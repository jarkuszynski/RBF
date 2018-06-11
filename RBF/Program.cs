﻿using AwokeKnowing.GnuplotCSharp;
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
            string fileTrain1 = "approximation_train_1.txt";
            string fileTrain2 = "approximation_train_2.txt";
            string fileTest = "approximation_test.txt";
            string filePathTrain1 = Path.GetFullPath(fileTrain1);
            string filePathTrain2 = Path.GetFullPath(fileTrain2);
            string filePathTest = Path.GetFullPath(fileTest);
            List<double> outputValues = new List<double>();



            DataGetter dataGetterTrain1 = new DataGetter(filePathTrain1);
            DataGetter dataGetterTrain2 = new DataGetter(filePathTrain2);
            DataGetter dataGetterTest = new DataGetter(filePathTest);



            double learnRate = 0.1;
            double momentum = 0.1;
            int numberOfRadialNeurons = 20;
            int numberOfEpochs = 300;



            NeuralNetwork neuralNetwork = new NeuralNetwork(dataGetterTrain1.getInputData(), learnRate, momentum, numberOfRadialNeurons);
            
            neuralNetwork.Train(numberOfEpochs, dataGetterTrain2);
            neuralNetwork.Train(numberOfEpochs, dataGetterTrain1);
            outputValues = neuralNetwork.Test(dataGetterTest);

            GnuPlot.Set("term wxt 0");
            GnuPlot.HoldOn();
            GnuPlot.Set("xlabel 'X Values'");
            GnuPlot.Set("ylabel 'Y Values'");
            GnuPlot.Plot(dataGetterTest.getInputData().ToArray(), dataGetterTest.getExpectedData().ToArray(),"title 'funkcja oryginalna'");
            GnuPlot.Plot(dataGetterTest.getInputData().ToArray(), outputValues.ToArray(),"title 'funkcja po aproksymacji'");
            GnuPlot.HoldOff();
            GnuPlot.Set("term wxt 1");
            GnuPlot.Set("xlabel 'Numer epoki'");
            GnuPlot.Set("ylabel 'Wartosc bledu'");
            GnuPlot.Plot(neuralNetwork.ErrorX.ToArray(), neuralNetwork.ErrorY.ToArray(),"with lines title 'Blad aproksymacji'");
            Console.ReadKey();
        }
    }
}
