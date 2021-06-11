namespace LostTech.TensorFlow.GPT {
    using System;
    using System.Collections.Generic;
    using System.Runtime.CompilerServices;

    public class GptHParams {
        public GptHParams(int embeddingDim, int attentionHeads, int encoderLayers, int contextTokens, int vocabularySize) {
            this.EmbeddingDim = embeddingDim;
            this.AttentionHeads = attentionHeads;
            this.EncoderLayers = encoderLayers;
            this.ContextTokens = contextTokens;
            this.VocabularySize = vocabularySize;
        }

        public GptHParams(IDictionary<string, int> hParams) {
            this.AttentionHeads = hParams.n_head();
            this.EncoderLayers = hParams.n_layer();
            this.ContextTokens = hParams.n_ctx();
            this.EmbeddingDim = hParams.n_embd();
            this.VocabularySize = hParams.n_vocab();
        }

        public int EmbeddingDim { get; }
        public int AttentionHeads { get; }
        public int EncoderLayers { get; }
        public int ContextTokens { get; }
        public int VocabularySize { get; }
    }

    public static class GptHParamsExtensions {
#pragma warning disable IDE1006 // Naming Styles
        public static int n_head(this IDictionary<string, int> hParams) => GetParam(hParams);
        public static int n_layer(this IDictionary<string, int> hParams) => GetParam(hParams);
        public static int n_embd(this IDictionary<string, int> hParams) => GetParam(hParams);
        public static int n_ctx(this IDictionary<string, int> hParams) => GetParam(hParams);
        public static int n_vocab(this IDictionary<string, int> hParams) => GetParam(hParams);
#pragma warning restore IDE1006 // Naming Styles

        static int GetParam(IDictionary<string, int> hParams, [CallerMemberName] string? name = null) => hParams[name ?? throw new ArgumentNullException(nameof(name))];
    }
}
