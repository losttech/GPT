namespace LostTech.TensorFlow.GPT {
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;

    using LostTech.Gradient;

    using tensorflow;

    using Xunit;

    public partial class GptTests {
        [Fact]
        public void Tune() {
            var hyperparams = new GptHParams(
                embeddingDim: 16,
                attentionHeads: 2,
                encoderLayers: 2,
                contextTokens: 16,
                vocabularySize: TestEncoder.Count);
            var encoder = new Gpt2Encoder(TestEncoder, TestBPE);
            var dataset = Gpt2Dataset.FromTexts(encoder, new[] { EncoderJson });

            var session = new Session();
            using var _ = session.StartUsing();

            int batchSize = 4;
            var input = tf.placeholder(tf.int32, new TensorShape(batchSize, null));
            var outputs = Gpt2Model.Model(hyperparams, input);
            var tuner = new Gpt2Tuner(hyperparams, session,
                                      inputPlaceholder: input,
                                      outputs,
                                      new GptTrainingSampler(dataset, new Random()),
                                      batchSize: batchSize);

            session.run(tf.global_variables_initializer());

            float loss0 = tuner.FineTuneOnBatch();
            float loss1 = tuner.FineTuneOnBatch();
            Assert.True(loss1 < loss0);
        }


        static readonly string EncoderJson = EmbededResources.ReadResource("encoder.json");
        static (string, string)[] TestBPE => BytePairEncoding.FromReader(new StringReader(EmbededResources.ReadResource("vocab.bpe"))).ToArray();
        static Dictionary<string, string> TestEncoder => Gpt2Encoder.LoadEncoderJson(EncoderJson);


    }
}
