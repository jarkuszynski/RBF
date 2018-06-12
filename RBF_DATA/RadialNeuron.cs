using System;
using System.Collections.Generic;

namespace RBF_DATA
{
    public class RadialNeuron
    {
        public double Centre { get; set; }
        public double Sigma { get; set; }
        public double OutputValue { get; set; }

        public RadialNeuron(double randomCentre)
        {
            Centre = randomCentre;
        }

        public double EuclideanDistance(RadialNeuron radialNeuron)
        {
            double result = 0;
            result = Math.Pow(Centre - radialNeuron.Centre, 2) * 1.0;
            return Math.Sqrt(result);
        }

        public double EuclideanDistance(double inputValue)
        {
            double result = 0;
            result = Math.Pow(inputValue - Centre, 2) * 1.0;
            return Math.Sqrt(result);
        }

        public void GaussianFunction(double inputValue)
        {
            double numerator = -1.0 * Math.Pow(EuclideanDistance(inputValue), 2) ;
            double denominator = 1.0 * 2 * Sigma * Sigma;
            OutputValue = Math.Exp((numerator / denominator) * 1.0) * 1.0;
        }

        public void UpdateSigma(List<RadialNeuron> radialNeurons)
        {
            double sum = 0;
            foreach (var radialNeuron in radialNeurons)
            {
                sum += EuclideanDistance(radialNeuron);
            }

            Sigma = Math.Sqrt((1.0 / radialNeurons.Count * 1.0) * sum * 1.0);
        }
    }
}