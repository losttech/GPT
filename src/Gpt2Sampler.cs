namespace LostTech.TensorFlow.GPT {
    using System;
    using System.Collections.Generic;

    using LostTech.Gradient;

    using tensorflow;
    using tensorflow.python.framework.dtypes;
    using tensorflow.python.ops.variable_scope;

    public static class Gpt2Sampler {
        static Tensor TopLogits(Tensor logits, int topK) {
            if (topK == 0)
                // no truncation
                return logits;

            Tensor TopK() {
                var valuesIndices = tf.nn.top_k(logits, k: topK);
                Tensor values = valuesIndices[0];
                Tensor minValues = values[.., ^1, tf.newaxis];
                return tf.where(logits < minValues,
                    tf.ones_like(logits, dtype: logits.dtype) * -1e10,
                    logits);
            }

            Tensor isTopKZero = tf.equal(topK, 0);
            return tf.cond(isTopKZero,
                true_fn: PythonFunctionContainer.Of(() => logits),
                false_fn: PythonFunctionContainer.Of(TopK));
        }

        public static Tensor SampleSequence(GptHParams hParams, int length,
            string startToken, int? batchSize = null,
            float temperature = 1, int topK = 0) {
            if (startToken is null) throw new ArgumentNullException(nameof(startToken));

            Tensor context = tf.fill_dyn(new[] { batchSize, 1 }, startToken);
            return SampleSequence(hParams, length, context, batchSize, temperature, topK);
        }
        public static Tensor SampleSequence(GptHParams hParams, int length,
            Tensor context, int? batchSize = null,
            float temperature = 1, int topK = 0) {
            if (hParams is null) throw new ArgumentNullException(nameof(hParams));
            if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));

            if (context is null)
                throw new ArgumentNullException(nameof(context));

            SortedDictionary<string, dynamic> Step(GptHParams @params, Tensor tokens, dynamic? past = null) {
                var lmOutput = Gpt2Model.Model(hParams: @params, input: tokens, past: past, reuse: _ReuseMode.AUTO_REUSE);

                var logits = lmOutput["logits"][.., .., ..@params.VocabularySize];
                Tensor presents = lmOutput["present"];
                int?[] pastShape = Gpt2Model.PastShape(hParams: @params, batchSize: batchSize);
                presents.set_shape_(new TensorShape(pastShape));

                return new SortedDictionary<string, object> {
                    ["logits"] = logits,
                    ["presents"] = presents,
                };
            }

            using var _ = new name_scope("sample_sequence").StartUsing();

            // Don't feed the last context token -- leave that to the loop below
            // TODO: Would be slightly faster if we called step on the entire context,
            // rather than leaving the last token transformer calculation to the while loop.
            var contextOutput = Step(hParams, context[.., ..^1]);

            Tensor[] Body(object? past, dynamic prev, object output) {
                var nextOutputs = Step(hParams, prev, past: past);
                Tensor logits = nextOutputs["logits"][.., ^1, ..] / tf.constant(temperature, dtypes.float32_ref);
                logits = TopLogits(logits, topK: topK);
                Tensor samples = tf.random.categorical(logits, num_samples: 1, dtype: tf.int32);
                return new Tensor[]
                {
                    past is null ? nextOutputs["presents"] : tf.concat(new []{ past, nextOutputs["presents"]}, axis: -2),
                    samples,
                    tf.concat(new []{ output, samples}, axis: 1),
                };
            }

            var vars = Body(null, context, context);

            bool True(object _a, object _b, object _c) => true;

            TensorShape[] shapeInvariants = new[]{
                new TensorShape(Gpt2Model.PastShape(hParams: hParams, batchSize: batchSize)),
                new TensorShape(batchSize, null),
                new TensorShape(batchSize, null),
            };

            Tensor maxTokens = tf.constant(length);
            // for some reason on CPU you can't sample longer texts
            // https://github.com/losttech/Gradient-Samples/issues/1
            if (!tf.test.is_gpu_available())
                maxTokens -= tf.shape(context)[1];

            Tensor result = tf.while_loop_dyn(
                cond: PythonFunctionContainer.Of<object, object, object, bool>(True),
                body: PythonFunctionContainer.Of(new Func<object, object, object, Tensor[]>(Body)),
                parallel_iterations: 10,
                swap_memory: false,
                name: null,
                maximum_iterations: length-1,
                loop_vars: vars,
                shape_invariants: shapeInvariants,
                back_prop: false)
                [2];
            return result;
        }
    }
}
