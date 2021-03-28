namespace LostTech.TensorFlow.GPT {
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;

    using numpy;

    using DataSet = System.Collections.Generic.List<numpy.ndarray>;

    public static class Gpt2Dataset {
        public static DataSet LoadDataset(Gpt2Encoder encoder, string path, string pattern = "*") {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException(nameof(path));
            var paths = new List<string>();
            if (Directory.Exists(path))
                paths.AddRange(Directory.EnumerateFiles(path, searchPattern: pattern, SearchOption.AllDirectories));
            else
                paths.Add(path);

            return LoadDataset(encoder, paths);
        }

        public static DataSet LoadDataset(Gpt2Encoder encoder, List<string> fileNames) {
            if (encoder is null) throw new ArgumentNullException(nameof(encoder));

            var tokenChunks = new DataSet();
            foreach (string file in fileNames) {
                Debug.WriteLine($"Reading {file}");
                if (Path.GetExtension(file) == ".npz") {
                    // pre-encoded
                    dynamic npzObject = np.load(file);
                    var npz = npzObject.__enter__();
                    foreach (var item in npz.files)
                        tokenChunks.Add(npz[item]);
                    npzObject.__exit__();
                } else {
                    string rawText = File.ReadAllText(file);
                    if (string.IsNullOrWhiteSpace(rawText))
                        continue;
                    var tokens = np.stack(encoder.Encode(rawText));
                    tokenChunks.Add(tokens);
                }
            }

            return tokenChunks;
        }

        const int TrimAfter = 16 * 1024 * 1024;

        public static DataSet FromTexts(Gpt2Encoder encoder, IEnumerable<string> texts) {
            var result = new DataSet();
            string encodedEndOfText = encoder.EncodedEndOfText;
            var chunk = new List<string>();
            int chunkSize = 0;
            void AddChunk() {
                var tokens = np.stack(chunk);
                chunk.Clear();
                chunkSize = 0;
                result.Add(tokens);
            }
            foreach (string text in texts) {
                if (string.IsNullOrWhiteSpace(text))
                    continue;

                if (chunkSize + text.Length + encodedEndOfText.Length >= TrimAfter) {
                    AddChunk();
                } else {
                    chunkSize += text.Length + encodedEndOfText.Length;
                    var encoded = encoder.Encode(text);
                    chunk.AddRange(encoded);
                    chunk.Add(encodedEndOfText);
                }
            }
            if (chunk.Count > 0)
                AddChunk();

            return result;
        }
    }
}
