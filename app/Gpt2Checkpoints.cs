namespace LostTech.TensorFlow.GPT {
    using System.IO;

    using tensorflow;

    public static class Gpt2Checkpoints {
        public const string CheckpointDir = "checkpoint";
        public const string Fresh = "fresh";
        public const string Latest = "latest";

        public static string GetLatestCheckpoint(string modelRoot, string? run) {
            string? latestCheckpoint = run is null
                ? null
                : tf.train.latest_checkpoint(Path.GetFullPath(Path.Combine(modelRoot, CheckpointDir, run)));
            latestCheckpoint ??= GetOriginalCheckpoint(modelRoot);
            return latestCheckpoint;
        }

        public static string GetOriginalCheckpoint(string modelRoot)
            => tf.train.latest_checkpoint(Path.GetFullPath(modelRoot));

        public static string ProcessCheckpointConfig(string modelRoot, string checkpoint, string? runName)
            => checkpoint switch {
                Latest => GetLatestCheckpoint(modelRoot: modelRoot, run: runName),
                Fresh => GetOriginalCheckpoint(modelRoot: modelRoot),
                _ => checkpoint,
            };
    }
}
