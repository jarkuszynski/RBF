using System;
using System.Collections.Generic;
using System.IO;

namespace RBF_DATA
{
    public class DataGetter
    {
        private List<double> inputData;
        private List<double> expectedData;

        public DataGetter(string filePath)
        {
            inputData = new List<double>();
            expectedData = new List<double>();

            var readLines = File.ReadAllLines(filePath);
            StreamReader streamReader = new StreamReader(filePath);
            do
            {
                string sr = streamReader.ReadLine();
                string[] parts = sr.Split(' ');
                inputData.Add(Convert.ToDouble(parts[0].Replace('.', ',')));
                expectedData.Add(Convert.ToDouble(parts[1].Replace('.', ',')));
                var ss = streamReader.EndOfStream;
            } while (!streamReader.EndOfStream);
        }

        public List<double> getInputData()
        {
            return inputData;
        }

        public List<double> getExpectedData()
        {
            return expectedData;
        }
    }
}