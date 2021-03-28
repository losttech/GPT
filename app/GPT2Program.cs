namespace LostTech.TensorFlow.GPT {
    using System;

    using LostTech.Gradient;
    using LostTech.TensorFlow;

    using ManyConsole.CommandLineUtils;

    static class Gpt2Program {
        static int Main(string[] args) {
            Console.Title = "GPT-2";
            GradientEngine.UseEnvironmentFromVariable();
            TensorFlowSetup.Instance.EnsureInitialized();

            return ConsoleCommandDispatcher.DispatchCommand(
                ConsoleCommandDispatcher.FindCommandsInSameAssemblyAs(typeof(Gpt2Program)),
                args, Console.Out);
        }
    }
}
