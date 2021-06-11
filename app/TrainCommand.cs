﻿namespace LostTech.TensorFlow.GPT {
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Text;
    using System.Threading;

    using ManyConsole.CommandLineUtils;

    using tensorflow;
    using tensorflow.core.protobuf.config_pb2;

    using DataSet = System.Collections.Generic.List<numpy.ndarray>;
    class TrainCommand : ConsoleCommand {
        public override int Run(string[] remainingArguments) {
            this.CheckRequiredArguments();
            if (remainingArguments.Length < 1)
                throw new ArgumentNullException("dataset");

            string modelPath = CommonCommandOptions.ExpandModelNameToPathOrExit(this.ModelName);

            string checkpoint = Gpt2Checkpoints.ProcessCheckpointConfig(
                modelPath: modelPath,
                checkpoint: this.Checkpoint,
                runName: this.RunName);

            var encoder = Gpt2Encoder.LoadEncoder(modelPath);

            string searchPattern = this.Include ?? "*";
            string datasetName = remainingArguments[0];
            var dataset = searchPattern.EndsWith("*.csv")
                ? LoadCsv(encoder, root: datasetName, field: this.ColumnName ?? throw new ArgumentException("column must be specified for training on .csv files"))
                : Gpt2Dataset.LoadDataset(encoder, path: datasetName, pattern: searchPattern);
            if (dataset.Count == 0) {
                Console.Error.WriteLine("The dataset is empty!");
                return -1;
            }

            var hParams = Gpt2Model.LoadHParams(modelPath);

            var random = this.Seed is null ? new Random() : new Random(this.Seed.Value);
            tf.random.set_seed(this.Seed);

            var stop = new CancellationTokenSource();
            Console.CancelKeyPress += delegate { stop.Cancel(); };

            dynamic config = config_pb2.ConfigProto.CreateInstance();
            config.gpu_options.allow_growth = true;
            var trainer = new Gpt2TunerLegacy(dataset, encoder, hParams, this.BatchSize, this.SampleLength, random) {
                SaveEvery = this.SaveEvery,
                SampleNum = this.SampleNum,
                SampleEvery = this.SampleEvery,
            };
            string checkpointOutputDirectory = Path.Combine(modelPath, Gpt2Checkpoints.CheckpointDir);
            trainer.FineTune(
                checkpointsDir: checkpointOutputDirectory, checkpoint: checkpoint,
                run: this.RunName,
                counter: checkpoint == "fresh" ? 1 : (int?)null,
                sessionConfig: config,
                cancellation: stop.Token);

            return 0;
        }

        static DataSet LoadCsv(Gpt2Encoder encoder, string root, string field) {
            var texts = new List<string>();
            var csvConfiguration = new CsvHelper.Configuration.Configuration {
                Delimiter = ",",
                HasHeaderRecord = true,
            };
            foreach (string file in Directory.EnumerateFiles(root, "*.csv", SearchOption.AllDirectories)) {
                using var reader = new CsvHelper.CsvReader(new StreamReader(file, Encoding.UTF8), csvConfiguration);
                reader.Read();
                reader.ReadHeader();
                while (reader.Read()) {
                    string entry = reader.GetField(field);
                    System.Diagnostics.Debug.Assert(reader.GetField(0).Length < 300);
                    if (!string.IsNullOrWhiteSpace(entry))
                        texts.Add(entry);
                }
            }
            return Gpt2Dataset.FromTexts(encoder, texts);
        }

        public string ModelName { get; set; } = "117M";
        public int? Seed { get; set; }
        public int BatchSize { get; set; } = 1;
        public int SampleLength { get; set; } = 1024;
        public int SampleNum { get; set; } = 1;
        public int SampleEvery { get; set; } = 1000;
        public int SaveEvery { get; set; } = 1000;
        public string RunName { get; set; } = DateTime.Now.ToString("s").Replace(':', '-');
        public string Checkpoint { get; set; } = "latest";
        public string? OutputDirectory { get; set; }
        public string? Include { get; set; }
        public string? ColumnName { get; set; }

        public TrainCommand() {
            this.IsCommand("train");
            this.HasAdditionalArguments(1, "<dataset>");
            this.HasOption("m|model=", "Which model to use", name => this.ModelName = name);
            this.HasOption("s|seed=",
                "Explicitly set seed for random generators to get reproducible results",
                (int s) => this.Seed = s);
            this.HasOption("i|include=", "Pattern of files to include in training",
                pattern => this.Include = pattern);
            this.HasOption("n|sample-num=", "",
                (int count) => this.SampleNum = count);
            this.HasOption("b|batch-size=", "Size of the batch, must divide sample-count",
                (int size) => this.BatchSize = size);
            this.HasOption("l|sample-length=", "",
                (int len) => this.SampleLength = len);
            this.HasOption("sample-every=", "Print a sample every N epochs",
                (int n) => this.SampleEvery = n);
            this.HasOption("save-every=", "How often to save a model, in epochs",
                (int n) => this.SaveEvery = n);
            this.HasOption("r|run=", "Name of the run (to be able to resume)",
                run => this.RunName = run);
            this.HasOption("c|checkpoint=", "Use specific checkpoint to start. Available values: 'latest' (default), 'fresh', or path to a checkpoint file",
                checkpoint => this.Checkpoint = checkpoint);
            this.HasOption("o|out-dir=", $"Place checkpoints into the specified directory (default: ./{Gpt2Checkpoints.CheckpointDir})",
                path => this.OutputDirectory = path);
            this.HasOption("column=", "Read texts from specific CSV column",
                name => this.ColumnName = name);
        }
    }
}
