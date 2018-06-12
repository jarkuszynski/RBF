using System;
using System.Collections.Generic;

namespace RBF_DATA
{
    public class Neuron
    {
        public List<double> Weights { get; set; }
        public List<double> DeltaWeights { get; set; }
        public double BiasWeight { get; set; }
        public double DeltaBiasWeight { get; set; }
        public double Gradient { get; set; }
        public double OutputValue { get; set; }

        public Neuron(Random random, List<RadialNeuron> radialNeurons)
        {
            Weights = new List<double>();
            DeltaWeights = new List<double>();
            for (int i = 0; i < radialNeurons.Count; i++)
            {
                Weights.Add((random.NextDouble() * 2.0) - 1.0);
                DeltaWeights.Add(0.0);
            }
            BiasWeight = (random.NextDouble() * 2.0) - 1.0;
        }

        public double CalculateOutputValue(List<RadialNeuron> radialNeurons)
        {
            double sum = 0;
            for (int i = 0; i < Weights.Count; i++)
            {
                sum += Weights[i] * radialNeurons[i].OutputValue;
            }
            sum += BiasWeight;
            OutputValue = LinearFunction.F(sum);
            return OutputValue;
        }

        public double CalculateError(double expectedDataSample)
        {
            return OutputValue - expectedDataSample;
        }

        public double CalculateGradient(double expectedDataSample)
        {
            Gradient = CalculateError(expectedDataSample) * LinearFunction.Derivative() * 1.0;
            return Gradient;
        }

        public void UpdateWeights(double learnRate, double momentum, List<RadialNeuron> radialNeurons)
        {
            double oldDeltaBiasWeight = DeltaBiasWeight;
            DeltaBiasWeight = -1.0 * learnRate * Gradient;
            BiasWeight += DeltaBiasWeight + momentum * oldDeltaBiasWeight * 1.0;

            for (int i = 0; i < Weights.Count; i++)
            {
                double oldDeltaWeight = DeltaWeights[i];
                DeltaWeights[i] = -1.0 * learnRate * Gradient * radialNeurons[i].OutputValue;
                Weights[i] += DeltaWeights[i] + momentum * oldDeltaWeight * 1.0;
            }
        }
    }
}