namespace LostTech.TensorFlow.GPT {
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using System.Text.Json;
    using LostTech.TensorFlow;
    using numpy;
    using MoreLinq;
    using Python.Runtime;
    using static System.Linq.Enumerable;
    using System.Globalization;

    public class Gpt2Encoder {
        public const string EndOfTextPseudoToken = "<|endoftext|>";

        readonly string errors;
        private readonly IDictionary<string, string> encoder;
        private readonly Dictionary<string, string> decoder;
        readonly Dictionary<byte, char> byteEncoder;
        readonly Dictionary<char, byte> byteDecoder;
        readonly Dictionary<(string, string), float> bpeRanks;
        static readonly dynamic regex, pattern;
        static readonly Dictionary<byte, char> bytesToUnicode = ComputeBytesToUnicode();

        static Gpt2Encoder() {
            TensorFlowSetup.Instance.EnsureInitialized();
            using var _ = Py.GIL();
            try {
                regex = Py.Import("regex");
            } catch (PythonException importError) {
                throw new FileNotFoundException(
                    "Python `regex` module is required. You may be able to install it using `pip install regex` command.",
                    importError);
            }

            pattern = regex.compile(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+");
        }

        public string EncodedEndOfText => this.encoder[EndOfTextPseudoToken];

        /// <summary>
        /// <para>     Returns list of utf-8 byte and a corresponding list of unicode strings.
        /// The reversible bpe codes work on unicode strings.
        /// This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        /// When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        /// </para>
        /// <para>
        /// This is a significant percentage of your normal, say, 32K bpe vocab.
        /// To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        /// </para>
        /// And avoids mapping to whitespace/control characters the bpe code barfs on.
        /// </summary>
        static Dictionary<byte, char> ComputeBytesToUnicode() {
            var bs = Range('!', '~' - '!' + 1)
                .Concat(Range('¡', '¬' - '¡' + 1))
                .Concat(Range('®', 'ÿ' - '®' + 1))
                .ToList();
            var cs = bs.ToList();
            int n = 0;
            foreach (int b in Range(0, 256)) {
                if (bs.Contains(b))
                    continue;

                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
            return bs.EquiZip(cs, (b, c) => ValueTuple.Create(checked((byte)b), checked((char)c))).ToDictionary();
        }
        /// <summary>
        /// Return set of symbol pairs in a word.
        /// </summary>
        /// <returns>Word is represented as tuple of symbols (symbols being variable-length strings).</returns>
        static ISet<(string, string)> GetPairs(string[] word) {
            var result = new SortedSet<(string, string)>();
            string prev = word[0];
            foreach (string symbol in word.Skip(1)) {
                result.Add((prev, symbol));
                prev = symbol;
            }
            return result;
        }

        public Gpt2Encoder(
            IDictionary<string, string> encoder,
            IEnumerable<(string, string)> bpeMerges,
            string errors = "replace") {
            this.encoder = encoder;
            this.decoder = encoder.ToDictionary(kv => kv.Value, kv => kv.Key);
            this.errors = errors;
            this.byteEncoder = bytesToUnicode;
            this.byteDecoder = this.byteEncoder.ToDictionary(kv => kv.Value, kv => kv.Key);
            this.bpeRanks = bpeMerges.Select((merge, index) => (merge, (float)index)).ToDictionary();
        }

        readonly Dictionary<string, string> cache = new();
        string BPE(string token) {
            if (this.cache.TryGetValue(token, out string result))
                return result;
            string[] word = token.Select(c => c.ToString()).ToArray();
            var pairs = GetPairs(word);
            if (pairs.Count == 0)
                return token;

            while (true) {
                var bigram = pairs.MinBy(pair => this.bpeRanks.GetValueOrDefault(pair, float.PositiveInfinity)).First();
                if (!this.bpeRanks.ContainsKey(bigram))
                    break;

                var (first, second) = bigram;
                var newWord = new List<string>();
                int i = 0;
                while (i < word.Length) {
                    int j = Array.IndexOf(word, first, startIndex: i);
                    if (j < 0) {
                        newWord.AddRange(word.Skip(i));
                        break;
                    }

                    newWord.AddRange(word.Skip(i).Take(j - i));
                    i = j;

                    if (word[i] == first && i < word.Length - 1 && word[i + 1] == second) {
                        newWord.Add(first + second);
                        i += 2;
                    } else {
                        newWord.Add(word[i]);
                        i++;
                    }
                }

                word = newWord.ToArray();
                if (word.Length == 1)
                    break;

                pairs = GetPairs(word);
            }

            result = string.Join(" ", word);
            this.cache[token] = result;
            return result;
        }

        public List<string> Encode(string text) {
            var bpeTokens = new List<string>();
            using var _ = Py.GIL();
            foreach (string token in regex.findall(pattern, text)) {
                string encoded = new(Encoding.UTF8.GetBytes(token)
                    .Select(@byte => this.byteEncoder[@byte]).ToArray());
                string bpe = this.BPE(encoded);
                foreach (string bpeToken in bpe.Split(' '))
                    bpeTokens.Add(this.encoder[bpeToken]);
            }
            return bpeTokens;
        }

        public string Decode(ndarray<int> tokens) {
            string[] tokenStrings = tokens.Cast<object>().Select(t => t.ToString()).ToArray();
            byte[] bytes = tokenStrings.SelectMany(token => this.decoder[token].Select(@char => this.byteDecoder[@char]))
                .ToArray();
            // TODO: error mode!
            return Encoding.UTF8.GetString(bytes);
        }

        public static Dictionary<string, string> LoadEncoderJson(string json)
            => JsonSerializer.Deserialize<Dictionary<string, int>>(json)
                .ToDictionary(kv => kv.Key, kv => kv.Value.ToString(CultureInfo.InvariantCulture));

        public static Gpt2Encoder LoadEncoder(string modelPath) {
            var encoder = LoadEncoderJson(File.ReadAllText(Path.Combine(modelPath, "encoder.json"), Encoding.UTF8));
            var bpeMerges = BytePairEncoding.FromFile(Path.Combine(modelPath, "vocab.bpe"));
            return new Gpt2Encoder(encoder, bpeMerges);
        }
    }
}
