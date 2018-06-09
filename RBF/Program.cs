using AwokeKnowing.GnuplotCSharp;
using RBF_DATA;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RBF
{
    class Program
    {
        static void Main(string[] args)
        {
            string file = "approximation_train_1.txt";
            string filePath = Path.GetFullPath(file);
            DataGetter dg = new DataGetter(filePath);
            NeuralNetwork neuralNetwork = new NeuralNetwork(dg, 0.1, 0.1, 20);
            neuralNetwork.Train(100);
            List<double> vs = new List<double>();
            for (int i = 0; i < neuralNetwork.OutputLayer.Count; i++)
            {
                vs.Add(neuralNetwork.OutputLayer[i].OutputValue);
            }
            GnuPlot.HoldOn();
            GnuPlot.Plot(neuralNetwork.InputData.ToArray(),neuralNetwork.Test().ToArray());
            GnuPlot.Plot(dg.getInputData().ToArray(), dg.getExpectedData().ToArray());
            GnuPlot.HoldOff();
            Console.ReadKey();
        }
    }
}
