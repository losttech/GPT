namespace LostTech.TensorFlow.GPT {
    using System.Collections.Generic;
    using System.IO;
    using System.Reflection;
    using System.Text;

    class EmbededResources {
        static readonly string Root = typeof(GptTests).Namespace + "._117M.";

        public static string ReadResource(string resourceName) {
            resourceName = Root + resourceName;
            var resourceContainer = Assembly.GetExecutingAssembly();
            using var stream = resourceContainer.GetManifestResourceStream(resourceName)!;
            if (stream is null) throw new KeyNotFoundException(resourceName);
            using var reader = new StreamReader(stream, Encoding.UTF8);
            return reader.ReadToEnd();
        }
    }
}
