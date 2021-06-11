namespace LostTech.TensorFlow.GPT {
    using System;
    using System.IO;
    class CommonCommandOptions {
        public static string? ExpandModelNameToPath(string modelName) {
            if (Directory.Exists(modelName))
                return Path.GetFullPath(modelName);
            else if (Directory.Exists(Path.Combine("models", modelName)))
                return Path.GetFullPath(Path.Combine("models", modelName));
            else
                return null;
        }

        public static string ExpandModelNameToPathOrExit(string modelName) {
            string? path = ExpandModelNameToPath(modelName);

            if (path is null) {
                Console.Error.WriteLine("model not found in " + Path.GetFullPath(modelName));
                Environment.Exit(-1);
                // should be unreachable
                throw new DirectoryNotFoundException("Model not found");
            }

            return path;
        }
    }
}
