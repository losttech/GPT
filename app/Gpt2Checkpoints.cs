namespace LostTech.TensorFlow.GPT {
    using System.IO;

    using tensorflow;

    public static class Gpt2Checkpoints {
        public const string CheckpointDir = "checkpoint";
        public const string Fresh = "fresh";
        public const string Latest = "latest";

        public static string GetLatestCheckpoint(string gpt2Root, string modelName, string? run) {
            string? latestCheckpoint = run is null
                ? null
                : tf.train.latest_checkpoint(Path.GetFullPath(Path.Combine(gpt2Root, CheckpointDir, run)));
            latestCheckpoint ??= GetOriginalCheckpoint(gpt2Root, modelName);
            return latestCheckpoint;
        }

        public static string GetOriginalCheckpoint(string gpt2Root, string modelName)
            => tf.train.latest_checkpoint(Path.GetFullPath(Path.Combine(gpt2Root, "models", modelName)));

        public static string ProcessCheckpointConfig(string gpt2Root, string checkpoint, string modelName, string? runName)
            => checkpoint switch {
                Latest => GetLatestCheckpoint(gpt2Root, modelName, runName),
                Fresh => GetOriginalCheckpoint(gpt2Root, modelName),
                _ => checkpoint,
            };
    }
}
