namespace LostTech.TensorFlow.GPT {
    using System.Runtime.CompilerServices;

    using LostTech.Gradient;

    using tensorflow.contrib.training;

    public class GptHParams: HParams {
        public GptHParams(int embeddingDim, int attentionHeads, int encoderLayers, int contextTokens, int vocabularySize)
            : base(kwargs: new {
                n_head = attentionHeads,
                n_layer = encoderLayers,
                n_ctx = contextTokens,
                n_embd = embeddingDim,
                n_vocab = vocabularySize,
            }.AsKwArgs()) {
            // TODO: validation
        }
    }

    public static class GptHParamsExtensions {
#pragma warning disable IDE1006 // Naming Styles
        public static int n_head(this IHParams hParams) => GetParam(hParams);
        public static int n_layer(this IHParams hParams) => GetParam(hParams);
        public static int n_embd(this IHParams hParams) => GetParam(hParams);
        public static int n_ctx(this IHParams hParams) => GetParam(hParams);
        public static int n_vocab(this IHParams hParams) => GetParam(hParams);
#pragma warning restore IDE1006 // Naming Styles

        static int GetParam(IHParams hParams, [CallerMemberName] string? name = null) => (int)hParams.get(name);
    }
}
