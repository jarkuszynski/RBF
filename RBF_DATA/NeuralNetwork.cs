using System;
using System.Collections.Generic;
using System.Linq;

namespace RBF_DATA
{
    public class NeuralNetwork
    {
        public List<double> ErrorX { get; set; }
        public List<double> ErrorY { get; set; }
        public DataGetter DataGetter { get; set; }
        public List<double> InputData { get; set; }
        public List<double> ExpectedData { get; set; }
        public List<Neuron> OutputLayer { get; set; }
        public List<RadialNeuron> HiddenLayer { get; set; }
        public double LearnRate { get; set; }
        public double Momentum { get; set; }

        public NeuralNetwork(List<double> inputData, double learnRate, double momentum, int numberOfRadialNeurons)
        {
            Random random = new Random();
            OutputLayer = new List<Neuron>();
            HiddenLayer = new List<RadialNeuron>();
            InputData = new List<double>();
            ExpectedData = new List<double>();

            LearnRate = learnRate;
            Momentum = momentum;
            InputData = inputData;
            int randomCentre;
            for (int i = 0; i < numberOfRadialNeurons; i++)
            {
                randomCentre = random.Next(0, InputData.Count);
                HiddenLayer.Add(new RadialNeuron(InputData[randomCentre]));
            }

            OutputLayer.Add(new Neuron(random, HiddenLayer));
        }

        public void Train(int numberOfEpochs, DataGetter dataGetter)
        {
            InputData = new List<double>();
            ExpectedData = new List<double>();
            ErrorX = new List<double>();
            ErrorY = new List<double>();
            InputData = dataGetter.getInputData();
            ExpectedData = dataGetter.getExpectedData();
            double error = 0;
            for (int i = 0; i < numberOfEpochs; i++)
            {
                error = 0;



                /* WYBOR SASIADUJACYCH NEURONOW DLA KAZDEGO NEURONU ABY ZAKTUALIZOWAC SIGME */
                foreach (var radialNeuron in HiddenLayer)
                {
                    var tempRadialNeurons = HiddenLayer.OrderBy(n => n.EuclideanDistance(radialNeuron)).Take(5).Skip(1).ToList();       //gdyby cos nie dzialalo warto to przeanaliozwac
                    radialNeuron.UpdateSigma(tempRadialNeurons);
                }

                for (int j = 0; j < InputData.Count; j++)
                {
                    /* FORWARD PROPAGATION - RADIAL NEURONS */
                    HiddenLayer.ForEach(rn => rn.GaussianFunction(InputData[j]));

                    /* FORWARD PROPAGATION - NEURONS */
                    OutputLayer.ForEach(n => n.CalculateOutputValue(HiddenLayer));

                    /* BACKPROPAGATION - NEURONS */
                    OutputLayer.ForEach(n => n.CalculateGradient(ExpectedData[j]));
                    OutputLayer.ForEach(n => n.UpdateWeights(LearnRate, Momentum, HiddenLayer));
                    error += MSE(ExpectedData[j]);
                }
                error = 1.0 * error / ExpectedData.Count;
                ErrorX.Add(i);
                ErrorY.Add(error);
                //if(i % (numberOfEpochs / 10.0) == 0)
                //    Console.WriteLine(error);
            }

            Console.WriteLine("Blad sredniokwadratowy aproksymacji: " + error + "\n");
        }

        public List<double> Test(DataGetter dataGetter)
        {
            InputData = new List<double>();
            ExpectedData = new List<double>();
            InputData = dataGetter.getInputData();
            ExpectedData = dataGetter.getExpectedData();
            double error = 0;
            List<double> vs = new List<double>();
            for (int i = 0; i < dataGetter.getInputData().Count; i++)
            {
                HiddenLayer.ForEach(rn => rn.GaussianFunction(dataGetter.getInputData()[i]));
                OutputLayer.ForEach(n => n.CalculateOutputValue(HiddenLayer));
                vs.Add(OutputLayer[0].OutputValue);
                error += MSE(ExpectedData[i]);
            }
            error = 1.0 * error / ExpectedData.Count;
            Console.WriteLine("Blad sredniokwadratowy aproksymacji: " + error);
            return vs;
        }

        public double MSE(double expectedDataSample)
        {
            double result = 0;
            result = OutputLayer.Sum(n => Math.Pow(n.CalculateError(expectedDataSample), 2));
            return result;
        }
    }
}