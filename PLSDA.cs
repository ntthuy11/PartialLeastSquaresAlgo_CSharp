using System;
using IM.Lib.Mathematics.Statistics;
using IM.Lib.Mathematics.Regression;

namespace IM.Lib.Classification
{
    /// <summary>
    /// Class to perform Partial Least Squares (PLS) discriminant analysis (DA).
    /// 
    /// <item> <term>1.01</term><description>(Thuy Nguyen) Add methods to support the regression matrix input from outside.</description> </item>
    /// <item> <term>1.0</term><description>(Thuy Nguyen) First release on Nov 1, 2013.</description> </item>
    /// </summary>
    /// 
    public class PLSDA
    {
        private PLSRegression plsR;

        private double[,] X;        // nSamples x nFeatures     // normalized
        private double[,] Y;        // nSamples x nClasses      // multipled

        private double[] meanOfX;   // 1 x nFeatures
        private double[] stdDevOfX; // 1 x nFeatures

        private int nSamples;       // from X or Y
        private int nFeatures;      // from X
        private int nClasses;       // from Y


        public double[,] GetNormalizedTrainingFeatures() { return this.X;           }
        public double[,] GetMultiTrainingClasses()       { return this.Y;           }

        public double[] GetMeanOfTrainingFeatures()      { return this.meanOfX;     }
        public double[] GetStdDevOfTrainingFeatures()    { return this.stdDevOfX;   }

        public int NSamples()   {   return this.nSamples;   }
        public int NFeatures()  {   return this.nFeatures;  }
        public int NClasses()   {   return this.nClasses;   }


        // --------------------------------------------------------------------


        /// <summary>Constructor: for the only use of 
        ///                     Classify(double[] features,  double[,] regressionMatrix)
        /// and                 Classify(double[,] features, double[,] regressionMatrix)
        /// </summary>
        public PLSDA()
        {

        }


        /// <summary>
        /// Constructor: input training features and classes to construct a PLSDA model
        /// 
        /// 1D array trainingClasses must be "multipled" to be used as an input of PLSDA. For example,
        ///     trainingClasses = {3, 3, 1, 2} 
        /// We have to convert it to
        ///     Y = { { 0, 0, 1 },
        ///           { 0, 0, 1 },
        ///           { 1, 0, 0 },
        ///           { 0, 1, 0 } }
        /// </summary>
        /// <param name="trainingFeatures">nSamples x nFeatures</param>
        /// <param name="trainingClasses">nSamples x 1</param>
        public PLSDA(double[,] trainingFeatures, int[] trainingClasses)
        {
            // calculate mean and standard deviation to normalize/centralize the trainingFeatures
            this.meanOfX   = MeanColumnWise(trainingFeatures);
            this.stdDevOfX = StdDevColumnWise(trainingFeatures, this.meanOfX);

            // normalize/centralize the trainingFeatures
            this.X = NormalizeX(trainingFeatures, this.meanOfX, this.stdDevOfX);
            this.nSamples  = X.GetUpperBound(0) + 1;
            this.nFeatures = X.GetUpperBound(1) + 1;            

            // prepare multiple-variable for each sample/label in the trainingClasses
            this.Y = PrepareMultipleVariablesForY(trainingClasses);
            this.nClasses = Y.GetUpperBound(1) + 1;

            // define PLS regression
            this.plsR = new PLSRegression(this.X, this.Y);
        }


        /// <summary> Create the PLSDA model by using PLSRegression</summary>
        /// <param name="nComponents">Number of PLS components</param>
        public void Train(int nComponents)
        {
            this.plsR.Train(nComponents);
        }


        /// <summary>Classify 1 input sample (1D array of feature)</summary>
        /// <param name="features">1 x nFeatures</param>
        /// <returns>The predicted class of the input</returns>
        public int Classify(double[] features)
        {
            return ClassifyOneSample(features, this.meanOfX, this.stdDevOfX, false, null);
        }


        /// <summary>
        /// Classify 1 input sample (1D array of feature). 
        /// This is used when we had the learned regressionMatrix B in advance.
        /// This method is independent from the input X and Y. We have to call "new PLSDA()" to use it.
        /// </summary>
        /// <param name="features">1 x nFeatures</param>
        /// <param name="regressionMatrix">(nFeatures + 1) x nClasses</param>
        /// <returns>The predicted class of the input</returns>
        public int Classify(double[] features, double[] meanX, double[] stdDevX, double[,] regressionMatrix)
        {
            return ClassifyOneSample(features, meanX, stdDevX, true, regressionMatrix);
        }


        /// <summary>Classify many input samples</summary>
        /// <param name="inputFeatures">nSamples x nFeatures</param>
        /// <returns>The predicted classes of the input</returns>
        public int[] Classify(double[,] features)
        {
            return ClassifyManySamples(features, this.meanOfX, this.stdDevOfX, false, null);
        }


        /// <summary>
        /// Classify many input samples. 
        /// This is used when we had the learned regressionMatrix B in advance.
        /// This method is independent from the input X and Y. We have to call "new PLSDA()" to use it.
        /// </summary>
        /// <param name="inputFeatures">nSamples x nFeatures</param>
        /// <param name="regressionMatrix">(nFeatures + 1) x nClasses</param>
        /// <returns>The predicted classes of the input</returns>
        public int[] Classify(double[,] features, double[] meanX, double[] stdDevX, double[,] regressionMatrix)
        {
            return ClassifyManySamples(features, meanX, stdDevX, true, regressionMatrix);
        }


        /// <summary>Measure how good this classification model is</summary>
        /// <param name="testingFeatures">nSamples x nFeatures</param>
        /// <param name="testingClasses">nSamples x 1</param>
        /// <returns>1D array of true (correct classification) and false (incorrect)</returns>
        public bool[] Validate(double[,] testingFeatures, int[] testingClasses)
        {            
            int[] classifiedClasses = Classify(testingFeatures);
            if(classifiedClasses.Length != testingClasses.Length)
                throw new Exception("Invalid input.");

            return ValidateWithClassifiedClasses(classifiedClasses, testingClasses);
        }


        /// <summary>Measure how good this classification model is</summary>
        /// <param name="classifiedClasses">nSamples x 1</param>
        /// <param name="testingClasses">nSamples x 1</param>
        /// <returns>1D array of true (correct classification) and false (incorrect)</returns>
        public bool[] ValidateWithClassifiedClasses(int[] classifiedClasses, int[] testingClasses)
        {
            if (classifiedClasses.Length != testingClasses.Length)
                throw new Exception("Invalid input.");

            bool[] result = new bool[testingClasses.Length];
            for (int i = 0; i < testingClasses.Length; i++)
                result[i] = (classifiedClasses[i] == testingClasses[i]);
            return result;
        }


        /// <summary>Calculate accuracy of classification</summary>
        /// <param name="testingFeatures">nSamples x nFeatures</param>
        /// <param name="testingClasses">nSamples x 1</param>
        /// <returns>Accuracy in percent</returns>
        public double CalculateAccuracy(double[,] testingFeatures, int[] testingClasses)
        {
            bool[] validationResult = Validate(testingFeatures, testingClasses);
            int count = 0;

            for (int i = 0; i < validationResult.Length; i++)
                if (validationResult[i])
                    count++;
            return count * 100.0 / validationResult.Length;
        }


        /// <summary>Calculate accuracy of classification</summary>
        /// <param name="classifiedClasses">nSamples x 1</param>
        /// <param name="testingClasses">nSamples x 1</param>
        /// <returns>Accuracy in percent</returns>
        public double CalculateAccuracyWithClassifiedClasses(int[] classifiedClasses, int[] testingClasses)
        {
            bool[] validationResult = ValidateWithClassifiedClasses(classifiedClasses, testingClasses);
            int count = 0;

            for (int i = 0; i < validationResult.Length; i++)
                if (validationResult[i])
                    count++;
            return count * 100.0 / validationResult.Length;
        }


        // ====================================== PRIVATE ======================================


        private int ClassifyOneSample(double[] features, double[] meanX, double[] stdDevX, bool isUseInputRegressionMatrix, double[,] regressionMatrix)
        {
            // convert the features from 1D array to 2D array to be as input for PLSRegression
            double[,] inputFeatures = new double[1, features.Length];
            for (int i = 0; i < features.Length; i++)
            {
                if (stdDevX[i] == 0)
                    inputFeatures[0, i] = 0;
                else
                    inputFeatures[0, i] = (features[i] - meanX[i]) / stdDevX[i]; // normalize the input
            }

            // predict the class
            double[,] predictedClass;
            if (isUseInputRegressionMatrix)
                predictedClass = new PLSRegression().PredictWithInputRegressionMatrixB(inputFeatures, regressionMatrix); // 1 x nClasses
            else
                predictedClass = this.plsR.Predict(inputFeatures); // 1 x nClasses

            // prepare the returned class for output 
            int theClass = 0;
            int nColsOfPredictedClass = predictedClass.GetUpperBound(1) + 1; // nClasses

            double minDiff = Math.Abs(predictedClass[0, 0] - 1);
            for (int i = 1; i < nColsOfPredictedClass; i++)
            {
                double diff = Math.Abs(predictedClass[0, i] - 1);
                if (minDiff > diff)
                {
                    minDiff = diff;
                    theClass = i;
                }
            }
            return theClass + 1; // because the real class begins from 1, but theClass begins from 0
        }


        private int[] ClassifyManySamples(double[,] features, double[] meanX, double[] stdDevX, bool isUseInputRegressionMatrix, double[,] regressionMatrix)
        {
            int nRowsOfFeatures = features.GetUpperBound(0) + 1; // nSamples
            int nColsOfFeatures = features.GetUpperBound(1) + 1; // nFeatures

            // normalize the input
            double[,] inputFeatures = new double[nRowsOfFeatures, nColsOfFeatures];
            for (int i = 0; i < nRowsOfFeatures; i++)
            {
                for (int j = 0; j < nColsOfFeatures; j++)
                {
                    if (stdDevX[j] == 0)
                        inputFeatures[i, j] = 0;
                    else
                        inputFeatures[i, j] = (features[i, j] - meanX[j]) / stdDevX[j];
                }
            }

            // predict the class
            double[,] predictedClasses;
            if (isUseInputRegressionMatrix)
                predictedClasses = new PLSRegression().PredictWithInputRegressionMatrixB(inputFeatures, regressionMatrix); // nSamples x nClasses
            else
                predictedClasses = this.plsR.Predict(inputFeatures); // nSamples x nClasses

            // prepare the returned classes for output 
            int[] theClasses = new int[nRowsOfFeatures];
            int nColsOfPredictedClasses = predictedClasses.GetUpperBound(1) + 1; // nClasses

            for (int j = 0; j < nRowsOfFeatures; j++) // check for every sample
            {
                double minDiff = Math.Abs(predictedClasses[j, 0] - 1);
                for (int i = 1; i < nColsOfPredictedClasses; i++) // find the "best" class
                {
                    double diff = Math.Abs(predictedClasses[j, i] - 1);
                    if (minDiff > diff)
                    {
                        minDiff = diff;
                        theClasses[j] = i;
                    }
                }
            }
            return Add(theClasses, 1); // plus 1 to theClasses (because the real class begins from 1, but theClass begins from 0)
        }


        // -----------------------------------------------------


        private double[] MeanColumnWise(double[,] input)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;

            double[] output = new double[nColsOfInput];
            for (int j = 0; j < nColsOfInput; j++)
            {
                double sum = 0;
                for (int i = 0; i < nRowsOfInput; i++)
                    sum += input[i, j];
                output[j] = sum / nRowsOfInput;
            }
            return output;
        }


        private double[] StdDevColumnWise(double[,] input, double[] meanOfColumnWise)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;

            double[] output = new double[nColsOfInput];
            for (int j = 0; j < nColsOfInput; j++)
            {
                double sum = 0;
                for (int i = 0; i < nRowsOfInput; i++)
                {
                    double deviation = input[i, j] - meanOfColumnWise[j];
                    sum += (deviation * deviation);
                }
                output[j] = Math.Sqrt(sum / nRowsOfInput);
            }
            return output;
        }


        private double[,] NormalizeX(double[,] X, double[] meanOfColumnWise, double[] stdDevOfColumnWise)
        {
            int nRows = X.GetUpperBound(0) + 1;
            int nCols = X.GetUpperBound(1) + 1;

            double[,] normalizedX = new double[nRows, nCols];
            for (int i = 0; i < nRows; i++)
                for (int j = 0; j < nCols; j++)
                    normalizedX[i, j] = (X[i, j] - meanOfColumnWise[j]) / stdDevOfColumnWise[j];
            return normalizedX;
        }


        private double[,] PrepareMultipleVariablesForY(int[] Y)
        {
            double[,] multiY = new double[Y.Length, FindMax(Y)];
            for (int i = 0; i < Y.Length; i++)
                multiY[i, Y[i] - 1] = 1.0;
            return multiY;
        }


        private int FindMax(int[] input)
        {
            int maxVal = input[0];
            for (int i = 1; i < input.Length; i++)
                if (maxVal < input[i])
                    maxVal = input[i];
            return maxVal;
        }


        private int[] Add(int[] input, int number)
        {
            int[] output = new int[input.Length];
            for (int i = 0; i < input.Length; i++)
                output[i] = input[i] + number;
            return output;
        }
    }
}
