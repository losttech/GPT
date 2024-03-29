﻿namespace LostTech.TensorFlow.GPT {
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Threading;

    using LostTech.Gradient;

    using ManyConsole.CommandLineUtils;

    using numpy;

    using tensorflow;
    using tensorflow.compat.v1;
    using tensorflow.compat.v1.train;
    using tensorflow.errors;

    class Gpt2Interactive : ConsoleCommand {
        /// <summary>
        /// Interactively run the model
        /// </summary>
        /// <param name="modelRoot">Which model to use</param>
        /// <param name="checkpoint">Which checkpoint to load</param>
        /// <param name="seed">Seed for random number generators, fix seed to reproduce results</param>
        /// <param name="sampleCount">Number of samples to return total</param>
        /// <param name="batchSize">Number of batches (only affects speed/memory).  Must divide sampleCount.</param>
        /// <param name="length">Number of tokens in generated text, if null (default), is
        ///     determined by model hyperparameters</param>
        /// <param name="temperature">randomness in boltzmann distribution.
        ///     Lower temperature results in less random completions. As the
        ///     temperature approaches zero, the model will become deterministic and
        ///     repetitive. Higher temperature results in more random completions.</param>
        /// <param name="topK">Controls diversity. 1 means only 1 word is
        ///     considered for each step (token), resulting in deterministic completions,
        ///     while 40 means 40 words are considered at each step. 0 (default) is a
        ///     special setting meaning no restrictions. 40 generally is a good value.
        /// </param>
        public static int Run(string modelRoot = "117M", string? checkpoint = null, int? seed = null,
            int sampleCount = 1,
            int batchSize = 1, int? length = null, float temperature = 1, int topK = 0) {
            if (sampleCount % batchSize != 0)
                throw new ArgumentException();

            var encoder = Gpt2Encoder.Load(modelRoot);
            var hParams = Gpt2Model.LoadHParams(modelRoot);

            int nCtx = hParams.ContextTokens;
            if (length is null)
                length = nCtx;
            else if (length > nCtx)
                throw new ArgumentException("Can't get samples longer than window size: " + nCtx);

            foreach(var gpu in tf.config.list_physical_devices("gpu"))
                tf.config.experimental.set_memory_growth(gpu, true);

            var sess = new Session(graph: new Graph());
            using var sessionContext = sess.StartUsing();

            Tensor context = v1.placeholder(tf.int32, new TensorShape(batchSize, null));
            tf.random.set_seed(seed);

            Tensor output = Gpt2Sampler.SampleSequence(
                hParams: hParams,
                length: length.Value,
                context: context,
                batchSize: batchSize,
                temperature: temperature,
                topK: topK);

            var saver = new Saver();
            checkpoint ??= tf.train.latest_checkpoint(modelRoot);
            saver.restore(sess, checkpoint);

            bool interrupted = false;
            Console.CancelKeyPress += (object sender, ConsoleCancelEventArgs args) =>
                Volatile.Write(ref interrupted, args.Cancel = true);

            while (!interrupted) {
                string text;
                do {
                    Console.Write("Model prompt >>> ");
                    text = Console.ReadLine();
                    if (Volatile.Read(ref interrupted)) break;
                    if (string.IsNullOrEmpty(text))
                        Console.WriteLine("Prompt should not be empty");
                } while (!Volatile.Read(ref interrupted) && string.IsNullOrEmpty(text));

                if (Volatile.Read(ref interrupted)) break;

                var contextTokens = encoder.Encode(text);
                if (!tf.test.is_gpu_available() && contextTokens.Count >= length.Value) {
                    Console.Error.WriteLine();
                    Console.Error.WriteLine("Prompt is too long.");
                    Console.Error.WriteLine();
                    continue;
                }
                int generated = 0;
                foreach (int _ in Enumerable.Range(0, sampleCount / batchSize)) {
                    ndarray<int> @out;
                    try {
                        @out = sess.run(output, feed_dict: new Dictionary<object, object> {
                            [context] = Enumerable.Repeat(contextTokens, batchSize).ToArray(),
                        })[.., contextTokens.Count..];
                    } catch (InvalidArgumentError ex) {
                        throw new ArgumentOutOfRangeException(
                            "Unable to generate sequence of desired length. "
                            + "Try lowering length by passing -l (-sample-length) parameter. "
                            + "Current length: " + length.Value,
                            innerException: ex);
                    }

                    foreach (int i in Enumerable.Range(0, batchSize)) {
                        generated++;
                        var part = @out[i].AsArray();
                        text = encoder.Decode(part);
                        Console.WriteLine($"{Delimiter} SAMPLE {generated} {Delimiter}");
                        Console.WriteLine(text);
                    }
                }
                Console.Write(Delimiter);
                Console.WriteLine(Delimiter);
            }

            return 0;
        }

        public Gpt2Interactive() {
            this.IsCommand("run");
            this.HasOption("m|model-path=", "Which model to use (directory path)", path => this.ModelPath = path);
            this.HasOption("s|seed=",
                "Explicitly set seed for random generators to get reproducible results",
                (int s) => this.Seed = s);
            this.HasOption("c|sample-count=", "Number of samples to generate for each prompt",
                (int count) => this.SampleCount = count);
            this.HasOption("b|batch-size=", "Size of the batch, must divide sample-count",
                (int size) => this.BatchSize = size);
            this.HasOption("l|sample-length=", "Length of the generated samples",
                (int len) => this.Length = len);
            this.HasOption("t|temperature=", "Randomness of the generated text",
                (float t) => this.Temperature = t);
            this.HasOption("k|top-k=", "Number of words to consider for each step",
                (int k) => this.TopK = k);
            this.HasOption("r|run=", "For tuned models, which run to use",
                run => this.RunName = run);
            this.HasOption("checkpoint=", "Which run checkpoint to use (default: latest)",
                checkpoint => this.Checkpoint = checkpoint);
        }

        public string ModelPath { get; set; } = Path.Combine("models", "117M");
        public int? Seed { get; set; }
        public int SampleCount { get; set; } = 1;
        public int BatchSize { get; set; } = 1;
        public int? Length { get; set; }
        public float Temperature { get; set; } = 1;
        public int TopK { get; set; }
        public string? RunName { get; set; }
        public string Checkpoint { get; set; } = "latest";

        public override int Run(string[] remainingArguments) {
            string checkpoint = Gpt2Checkpoints.ProcessCheckpointConfig(
                modelRoot: this.ModelPath,
                checkpoint: this.Checkpoint,
                runName: this.RunName);

            return Run(
                modelRoot: this.ModelPath,
                checkpoint: checkpoint,
                seed: this.Seed,
                sampleCount: this.SampleCount,
                batchSize: this.BatchSize,
                length: this.Length,
                temperature: this.Temperature,
                topK: this.TopK);
        }

        const string Delimiter = "========================================";
    }
}
