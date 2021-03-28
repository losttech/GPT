namespace LostTech.TensorFlow.GPT {
    using numpy;
    public interface IGptTrainingSampleGenerator {
        ndarray Sample(int length);
    }
}
