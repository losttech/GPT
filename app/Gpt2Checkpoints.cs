namespace LostTech.TensorFlow.GPT {
    using System.IO;

    using tensorflow;

    public static class Gpt2Checkpoints {
        public const string CheckpointDir = "checkpoint";
        public const string Fresh = "fresh";
        public const string Latest = "latest";

        public static string GetLatestCheckpoint(string modelPath, string? run) {
            string? latestCheckpoint = run is null
                ? null
                : tf.train.latest_checkpoint(Path.GetFullPath(Path.Combine(modelPath, CheckpointDir, run)));
            latestCheckpoint ??= GetOriginalCheckpoint(modelPath);
            return latestCheckpoint;
        }

        public static string GetOriginalCheckpoint(string modelPath)
            => tf.train.latest_checkpoint(Path.GetFullPath(modelPath));

        public static string ProcessCheckpointConfig(string checkpoint, string modelPath, string? runName)
            => checkpoint switch {
                Latest => GetLatestCheckpoint(modelPath, runName),
                Fresh => GetOriginalCheckpoint(modelPath),
                _ => checkpoint,
            };
    }
}
