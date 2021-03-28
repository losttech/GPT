namespace LostTech.TensorFlow.GPT {
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;

    // TODO: full managed BPE implementation
    public static class BytePairEncoding {
        public static IEnumerable<(string, string)> Load(IEnumerable<string> vocabulary)
            => vocabulary.Select(merge => ParseMergeEntry(merge));

        public static (string, string)[] FromFile(string path) {
            using var reader = new StreamReader(path, Encoding.UTF8);
            return FromReader(reader).ToArray();
        }
        public static IEnumerable<(string, string)> FromReader(TextReader reader)
            => Load(VocabularyEntries(reader));
        
        static IEnumerable<string> VocabularyEntries(TextReader reader) {
            if (reader is null) throw new ArgumentNullException(nameof(reader));

            string? line = reader.ReadLine();
            if (string.IsNullOrEmpty(line)) throw new FormatException();

            var entry = ParseMergeEntry(line);
            if (entry.Item1 == "#version:") {
                if (entry.Item2 != "0.2") throw new NotSupportedException(line);
                line = reader.ReadLine();
            }

            while (line != null) {
                if (!string.IsNullOrEmpty(line)) {
                    yield return line;
                }

                line = reader.ReadLine();
            }
        }

        static (string, string) ParseMergeEntry(string merge)
            => (merge.Split(' ')[0], merge.Split(' ')[1]);
    }
}
