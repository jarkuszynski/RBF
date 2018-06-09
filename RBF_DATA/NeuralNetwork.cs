using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RBF_DATA
{
    public class NeuralNetwork
    {
        public DataGetter DataGetter { get; set; }
        public List<double> InputData { get; set; }
        public List<double> ExpectedData { get; set; }
        public List<Neuron> OutputLayer { get; set; }
        public List<RadialNeuron> HiddenLayer { get; set; }
        public double LearnRate { get; set; }
        public double Momentum { get; set; }

        public NeuralNetwork(DataGetter dataGetter, double learnRate, double momentum, int numberOfRadialNeurons)
        {
            Random random = new Random();
            OutputLayer = new List<Neuron>();
            HiddenLayer = new List<RadialNeuron>();
            InputData = new List<double>();
            ExpectedData = new List<double>();

            DataGetter = dataGetter;
            LearnRate = learnRate;
            Momentum = momentum;
            InputData = DataGetter.getInputData();
            ExpectedData = DataGetter.getExpectedData();
            int randomCentre;
            for (int i = 0; i < numberOfRadialNeurons; i++)
            {
                randomCentre = random.Next(0, InputData.Count);
                HiddenLayer.Add(new RadialNeuron(InputData[randomCentre]));
            }

            OutputLayer.Add(new Neuron(random, HiddenLayer));
        }

        public void Train(int numberOfEpochs)
        {
            List<RadialNeuron> EndRadialNeurons = new List<RadialNeuron>(); ;
            for (int i = 0; i < numberOfEpochs; i++)
            {
                for (int j = 0; j < InputData.Count; j++)
                {
                    /* WYBOR SASIADUJACYCH NEURONOW DLA KAZDEGO NEURONU ABY ZAKTUALIZOWAC SIGME */
                    foreach (var radialNeuron in HiddenLayer)
                    {
                        var tempRadialNeurons = HiddenLayer.OrderBy(n => n.EuclideanDistance(radialNeuron)).Take(5).Skip(1).ToList();       //gdyby cos nie dzialalo warto to przeanaliozwac
                        radialNeuron.UpdateSigma(tempRadialNeurons);
                    }

                    /* FORWARD PROPAGATION - RADIAL NEURONS */
                    HiddenLayer.ForEach(rn => rn.GaussianFunction(InputData[j]));

                    /* FORWARD PROPAGATION - NEURONS */
                    OutputLayer.ForEach(n => n.CalculateOutputValue(HiddenLayer));

                    /* BACKPROPAGATION - NEURONS */
                    OutputLayer.ForEach(n => n.CalculateGradient(ExpectedData[j]));
                    OutputLayer.ForEach(n => n.UpdateWeights(LearnRate, Momentum, HiddenLayer));
                }
            }
        }

        public List<double> Test()
        {
            List<double> vs = new List<double>();
            for (int i = 0; i < InputData.Count; i++)
            {
                HiddenLayer.ForEach(rn => rn.GaussianFunction(InputData[i]));
                OutputLayer.ForEach(n => n.CalculateOutputValue(HiddenLayer));
                vs.Add( OutputLayer[0].OutputValue);
            }
            return vs;
        }
    }
}
