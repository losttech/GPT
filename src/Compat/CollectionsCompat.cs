// ReSharper disable once CheckNamespace
namespace System.Collections.Generic {
    static class CollectionsCompat {
        public static TValue GetValueOrDefault<TKey, TValue>(
            this IReadOnlyDictionary<TKey, TValue> dict,
            TKey key, TValue @default) {
            if (dict is null) throw new ArgumentNullException(nameof(dict));

            return dict.TryGetValue(key, out var result) ? result : @default;
        }
    }
}
