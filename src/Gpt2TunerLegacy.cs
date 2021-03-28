// ported from https://github.com/nshepperd/gpt-2

namespace LostTech.TensorFlow.GPT {
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Threading;

    using LostTech.Gradient;

    using numpy;

    using tensorflow;
    using tensorflow.contrib.training;
    using tensorflow.train;

    using static System.FormattableString;

    using DataSet = System.Collections.Generic.List<numpy.ndarray>;

    [Obsolete("Use Gpt2Tuner instead")]
    public class Gpt2TunerLegacy {
        const string SampleDir = "samples";

        readonly DataSet dataset;
        readonly Gpt2Encoder encoder;
        readonly IHParams hParams;
        readonly int batchSize;
        readonly int sampleLength;
        readonly Random random;

        public Gpt2TunerLegacy(DataSet dataset, Gpt2Encoder encoder, IHParams hParams,
            int batchSize, int sampleLength, Random random) {
            this.dataset = dataset ?? throw new ArgumentNullException(nameof(dataset));
            this.encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
            this.hParams = hParams ?? throw new ArgumentNullException(nameof(hParams));
            this.batchSize = batchSize;
            this.sampleLength = sampleLength;
            this.random = random ?? throw new ArgumentNullException(nameof(random));
        }

        public int SaveEvery { get; set; } = 1000;
        public int SampleEvery { get; set; } = 100;
        public int SampleNum { get; set; } = 1;

        public void FineTune(string checkpointsDir, string checkpoint, string run, int? counter,
                             int topK = 40, float temperature = 1.0f,
                             dynamic? sessionConfig = null, CancellationToken cancellation = default) {
            Session session = sessionConfig is null
                ? Session.NewDyn(config: sessionConfig)
                : new Session();
            using var _ = session.StartUsing();

            Tensor context = tf.placeholder(tf.int32, new TensorShape(this.batchSize, null));
            var output = Gpt2Model.Model(this.hParams, input: context);

            var sampler = new GptTrainingSampler(this.dataset, this.random);
            var optimizer = new AdamOptimizer(learning_rate: 0.0002);
            var tuner = new Gpt2Tuner(this.hParams, session, context, output, sampler, this.batchSize, optimizer);

            Tensor sample = Gpt2Sampler.SampleSequence(
                this.hParams,
                length: this.sampleLength,
                context: context,
                batchSize: this.batchSize,
                temperature: temperature,
                topK: topK);

            var saver = new Saver(
                var_list: tuner.ModelVariables,
                max_to_keep: 5,
                keep_checkpoint_every_n_hours: 1);

            session.run(tf.global_variables_initializer());

            Console.WriteLine("Loading checkpoint " + checkpoint);
            saver.restore(session, checkpoint);

            Console.WriteLine("Loading dataset...");
            
            Console.WriteLine($"Dataset has {sampler.TokenCount} tokens");

            string counterFile = Path.Combine(checkpointsDir, run, "counter");
            if (counter is null && File.Exists(counterFile))
                counter = int.Parse(File.ReadAllText(counterFile), CultureInfo.InvariantCulture) + 1;
            counter ??= 1;

            string runCheckpointDir = Path.Combine(checkpointsDir, run);
            string runSampleDir = Path.Combine(SampleDir, run);

            void Save() {
                Directory.CreateDirectory(runCheckpointDir);
                Console.WriteLine("Saving " + Path.Combine(runCheckpointDir, Invariant($"model-{counter}")));
                saver.save(session,
                    Path.Combine(runCheckpointDir, "model"),
                    global_step: counter.Value);
                File.WriteAllText(path: counterFile, contents: Invariant($"{counter}"));
            }

            void GenerateSamples() {
                var contextTokens = np.array(new[] { this.encoder.EncodedEndOfText });
                var allText = new List<string>();
                int index = 0;
                string? text = null;
                while (index < this.SampleNum) {
                    ndarray<int> @out = session.run(sample, feed_dict: new Dictionary<object, object> {
                        [context] = Enumerable.Repeat(contextTokens, this.batchSize),
                    });
                    foreach (int i in Enumerable.Range(0, Math.Min(this.SampleNum - index, this.batchSize))) {
                        text = this.encoder.Decode((ndarray<int>)@out[i]);
                        text = Invariant($"======== SAMPLE {index + 1} ========\n{text}\n");
                        allText.Add(text);
                        index++;
                    }
                }
                Console.WriteLine(text);
                Directory.CreateDirectory(runSampleDir);
                File.WriteAllLines(
                    path: Path.Combine(runSampleDir, Invariant($"samples-{counter}")),
                    contents: allText);
            }

            var avgLoss = (0.0, 0.0);
            var startTime = DateTime.Now;

            while (!cancellation.IsCancellationRequested) {
                if (counter % this.SaveEvery == 0)
                    Save();
                if (counter % this.SampleEvery == 0)
                    GenerateSamples();

                float lv = tuner.FineTuneOnBatch();

                avgLoss = (avgLoss.Item1 * 0.99 + lv, avgLoss.Item2 * 0.99 + 1);

                Console.WriteLine($"[{counter} | {DateTime.Now - startTime}] loss={lv} avg={avgLoss.Item1 / avgLoss.Item2}");

                counter++;
            }

            Console.WriteLine("Interrupted");
            Save();
        }
    }
}
