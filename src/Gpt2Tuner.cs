// ported from https://github.com/nshepperd/gpt-2

namespace LostTech.TensorFlow.GPT {
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using LostTech.Gradient.BuiltIns;

    using MoreLinq;

    using tensorflow;
    using tensorflow.compat.v1;
    using tensorflow.compat.v1.train;
    using Variable = tensorflow.Variable;

    public class Gpt2Tuner {
        public GptHParams Hyperparams { get; }
        readonly ISession session;
        public Tensor InputPlaceholder { get; }
        readonly Dictionary<string, Tensor> outputs;
        readonly Operation optimizerStep;
        public IOptimizer Optimizer { get; }
        public Tensor Loss { get; }
        public IGptTrainingSampleGenerator Sampler { get; }
        public int BatchSize { get; }
        public IReadOnlyList<Variable> ModelVariables { get; }

        public Gpt2Tuner(GptHParams hyperparams, ISession session,
                         Tensor inputPlaceholder, Dictionary<string, Tensor> outputs,
                         IGptTrainingSampleGenerator sampler,
                         int batchSize,
                         IOptimizer? optimizer = null) {
            this.Hyperparams = hyperparams ?? throw new ArgumentNullException(nameof(hyperparams));
            this.session = session ?? throw new ArgumentNullException(nameof(session));
            this.InputPlaceholder = inputPlaceholder ?? throw new ArgumentNullException(nameof(inputPlaceholder));
            this.outputs = outputs ?? throw new ArgumentNullException(nameof(outputs));
            this.Sampler = sampler ?? throw new ArgumentNullException(nameof(sampler));
            this.BatchSize = batchSize;

            this.Optimizer = optimizer ?? new AdamOptimizer(learning_rate: 0.0002);

            this.ModelVariables = Enumerable.Where((PythonList<Variable>)v1.trainable_variables(), var => var.name.Contains("model")).ToArray();

            Tensor labels = inputPlaceholder[.., 1..];
            Tensor logits = outputs["logits"][.., ..^1];
            this.Loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits_dyn(
                    labels: labels,
                    logits: logits));

            this.optimizerStep = this.Optimizer.minimize(this.Loss, var_list: this.ModelVariables);
        }

        /// <returns>Loss</returns>
        public float FineTuneOnBatch() {
            var batch = MoreEnumerable
                .GenerateByIndex(_ => this.Sampler.Sample(this.Hyperparams.ContextTokens))
                .Take(this.BatchSize)
                .ToArray();

            var placeholderValues = new Dictionary<object, object> {
                [this.InputPlaceholder] = batch,
            };
            var tuple = this.session.run((this.optimizerStep, this.Loss), feed_dict: placeholderValues);

            return tuple[1];
        }
    }
}
