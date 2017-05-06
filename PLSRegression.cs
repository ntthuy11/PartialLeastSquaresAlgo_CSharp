using System;

namespace IM.Lib.Mathematics.Regression
{
    /// <summary>
    /// Class to perform Partial Least Squares (PLS) regression.
    /// The idea was originated from http://www.mathworks.com/matlabcentral/fileexchange/30685-pls-regression-or-discriminant-analysis-with-leave-one-out-cross-validation-and-prediction
    /// 
    /// The implementation is followed this paper 
    ///     Sijimen de Jong, "SIMPLS: an alternative approach to partial least squares regression," 
    ///                     Chemometrics and Intelligent Laboratory Systems, 18 (1993) 251-263
    /// 
    /// <item> <term>1.0</term><description>(Thuy Nguyen) First release on Oct 31, 2013.</description> </item>
    /// </summary>
    /// 
    public class PLSRegression
    {
        private double[,] X;    // X = TP' + E              // nSamples  x nFeatures
        private double[,] Y;    // Y = UQ' + F              // nSamples  x nDimY

        private double[,] T;    // score matrix of X        // nSamples  x nComponents
        private double[,] U;    // score matrix of Y        // nSamples  x nComponents
        private double[,] P;    // loading matrix of X      // nFeatures x nComponents
        private double[,] Q;    // loading matrix of Y      // nDimY     x nComponents

        private double[,] W;    // weight                   // nFeatures x nComponents
        private double[,] B;    // regression matrix        // (nFeatures + 1) x nDimY   // the first row is for intercept
        private double[,] B0;   // intercept of X and Y     // 1 x nDimY

        private int nSamples;   // from X or Y
        private int nFeatures;  // from X
        private int nDimY;      // from Y


        public double[,] GetT_scoreMatrixOfX()      {   return this.T;      }        
        public double[,] GetU_scoreMatrixOfY()      {   return this.U;      }
        public double[,] GetP_loadingMatrixOfX()    {   return this.P;      }
        public double[,] GetQ_loadingMatrixOfY()    {   return this.Q;      } 
        public double[,] GetW_weightMatrix()        {   return this.W;      }
        public double[,] GetB_regressionMatrix()    {   return this.B;      }
        public double[,] GetB0_intercept()          {   return this.B0;     }

        public int NSamples()   {   return this.nSamples;   }
        public int NFeatures()  {   return this.nFeatures;  }
        public int NDimY()      {   return this.nDimY;      }


        // --------------------------------------------------------------------


        /// <summary>Constructor: for the only use of PredictWithInputRegressionMatrixB</summary>
        public PLSRegression()
        {

        }


        /// <summary>Constructor: input X and Y to construct a PLS model</summary>
        /// <param name="X">nSamples x nFeatures</param>
        /// <param name="Y">nSamples x nDimY</param>
        public PLSRegression(double[,] X, double[,] Y)
        {
            this.X = X;
            this.Y = Y;

            // check the input
            if (CheckInput(X, Y))
            {
                this.nSamples   = X.GetUpperBound(0) + 1;
                this.nFeatures  = X.GetUpperBound(1) + 1;
                this.nDimY      = Y.GetUpperBound(1) + 1; // nDim = nClass in case of classification
            }
        }


        /// <summary>
        /// Create the PLS model. This will compute matrices of score, loading, weight, regression for X and Y. <br/>
        /// Use the getters to get these matrices: T (nSamples x nComponents), U (nSamples x nComponents), P (nFeatures x nComponents), 
        ///                                        Q (nDimY x nComponents), W (nFeatures x nComponents), B ((nFeatures + 1) x nDimY), B0 (1 x nDimY)
        /// </summary>
        /// <param name="nComponents">Number of PLS components</param>
        public void Train(int nComponents) // nComponents = nLatentVariables
        {
            // initialize T, U, P, Q, W
            this.T = new double[this.nSamples,      nComponents];
            this.U = new double[this.nSamples,      nComponents];
            this.P = new double[this.nFeatures,     nComponents];
            this.Q = new double[this.nDimY,         nComponents];
            this.W = new double[this.nFeatures,     nComponents];
            this.B = new double[this.nFeatures + 1, this.nDimY];

            double[,] V = new double[this.nFeatures, nComponents]; // for deflate 
            // (DEFLATE: to remove the already determined solution, while leaving the remainder solutions unchanged)

            // 0. cross-product of X and Y
            double[,] Xtranpose = Transpose(X);         // nFeatures x nSamples
            double[,] XY = CrossProduct(Xtranpose, Y);  // nFeatures x nDimY
            
            // iteration
            for (int iComponent = 0; iComponent < nComponents; iComponent++)
            {
                // 1. singular value decomposition 
                double[,] A, Ct;
                double[] B; 
                SVD(XY, out A, out B, out Ct);        // A (nFeatures x nDimY), Ct (nDimY x nDimY)
                                                      // B (a diagonal matrix, nDimY x nDimY, which diagonal elements are singular values)

                // 2. initialize the loadings s0 and dt0 for X and Y (based on SVD)
                double[,] a0  = new double[this.nFeatures, 1];      CopyOneColumn(A, 0, ref a0, 0);                 // a0 = A(:, 0)
                double b0     = B[0];                                                                               // b0 = B(0)
                double[,] c0  = new double[this.nDimY, 1];          CopyOneColumn(Transpose(Ct), 0, ref c0, 0);     // c0 = C(0)

                // 3. calculate the score of X
                double[,] t0  = CrossProduct(X, a0);            // score vector (nSamples x 1) of X                 // t0 = X*a0
                double t0norm = Norm(t0);                       // norm of the score vector                         
                t0            = Divide(t0, t0norm);             // normalize the score vector                       // t0 = t0 ./ ||t0||
                CopyOneColumn(t0, 0, ref this.T, iComponent);   // copy the score vector t0 to the score matrix T   // T(:, i) = t0;

                // 4. calculate the loading of X
                double[,] p0  = CrossProduct(Xtranpose, t0);    // loading vector (nFeatures x 1) of X              // p0 = X' * t0
                CopyOneColumn(p0, 0, ref this.P, iComponent);   // copy the loading vector to the loading matrix P  // P(:, i) = p0

                // 5. calculate the loading of Y
                double[,] q0  = Multiply(Divide(c0, t0norm), b0); // loading of Y                                   // q0 = b0 * ct0 / ||t0||
                CopyOneColumn(q0, 0, ref this.Q, iComponent);   // loading matrix Q                                 // Q(:, i) = q0

                // 6. calculate the score of Y
                double[,] u0; 
                if (this.nDimY == 1) u0 = Multiply(Y, q0[0, 0]);// score of Y                                       // u0 = Y*q0
                else                 u0 = CrossProduct(Y, q0);
                CopyOneColumn(u0, 0, ref this.U, iComponent);   // score matrix U                                   // U(:, i) = u0

                // 7. calculate the weight
                double[,] w0  = Divide(a0, t0norm);             // weight vector                                    // w0 = a0 ./ ||t0||
                CopyOneColumn(w0, 0, ref this.W, iComponent);   // weight matrix W                                  // W(:, i) = w0

                // 8. deflate the matrix XY
                DeflateXY(iComponent, ref V, ref XY);
            }

            // 9. calculate the regression matrix B
            CalculateB_regressionMatrix();

            // ADDTIONAL: 10. deflate U (for better U)
            DeflateU(nComponents);
        }


        /// <summary>Get the percentage of variance explained by X</summary>
        /// <returns>Array of 1 x nComponents</returns>
        public double[,] GetPercentageOfVarianceExplainedByX()
        {
            // sum(P.^2, 1) ./ sum(sum(x.^2, 1))
            double[,] numerator = SumColumnWise(Square(this.P));       // 1 x nComponents                              // sum(P.^2, 1)
            double denominator  = Sum(SumColumnWise(Square(this.X)));                                                  // sum(sum(x.^2, 1))
            double[,] result    = Divide(numerator, denominator);

            // multiply by 100 to have the percentage
            return Multiply(result, 100);
        }


        /// <summary>Get the percentage of variance explained by X</summary>
        /// <returns>Array of 1 x nComponents</returns>
        public double[,] GetPercentageOfVarianceExplainedByY()
        {
            // sum(Q.^2, 1) ./ sum(sum(y.^2, 1))
            double[,] numerator = SumColumnWise(Square(this.Q));       // 1 x nComponents                              // sum(Q.^2, 1)
            double denominator  = Sum(SumColumnWise(Square(this.Y)));                                                  // sum(sum(Y.^2, 1))
            double[,] result    = Divide(numerator, denominator);

            // multiply by 100 to have the percentage
            return Multiply(result, 100);
        }


        /// <summary>Predict Y according to the input X</summary>
        /// <param name="X0">nSamples x nFeatures</param>
        /// <returns>Predicted Y</returns>
        public double[,] Predict(double[,] X0)
        {
            return PredictWithInputRegressionMatrixB(X0, this.B);
        }


        /// <summary>
        /// Predict Y according to the input X, with the input regression matrix B.
        /// This method is independent from the input X and Y. We have to call "new PLSRegression()" to use it.
        /// </summary>
        /// <param name="X0">nSamples x nFeatures</param>
        /// <param name="inputB">(nFeatures + 1) x nDimY</param>
        /// <returns>Predicted Y</returns>
        public double[,] PredictWithInputRegressionMatrixB(double[,] X0, double[,] inputB)
        {
            int nRowsOfX0 = X0.GetUpperBound(0) + 1; // nSamples
            int nColsOfX0 = X0.GetUpperBound(1) + 1; // nFeatures

            // prepare X1[0; X0]
            double[,] X1 = new double[nRowsOfX0, 1 + nColsOfX0];
            for (int i = 0; i < nRowsOfX0; i++) X1[i, 0] = 1; // assign 1 (s) to the first column of X1 (for the intercept)

            // copy the whole X0 to X1: 1 -> end
            CopyColumns(X0, 0, nColsOfX0 - 1, ref X1, 1);

            return CrossProduct(X1, inputB);
        }


        /// <summary>Validate the PLS model using the input X and Y by measuring the root mean square (RMS) error</summary>
        /// <param name="Xtest">nSamples x nFeatures</param>
        /// <param name="Ytest">nSamples x nDimY</param>
        /// <returns>RMS errors (nSamples x 1) of all samples in Y</returns>
        public double[,] ValidateUsingRMS(double[,] Xtest, double[,] Ytest) // calculate the error of the PLS model using root mean squares
        {
            return ValidateUsingRMSwithPredictedY(Predict(Xtest), Ytest);
        }


        /// <summary>Validate the PLS model using the input X and Y by measuring the root mean square (RMS) error</summary>
        /// <param name="Ypredict">nSamples x nDimY</param>
        /// <param name="Ytest">nSamples x nDimY</param>
        /// <returns>RMS errors (nSamples x 1) of all samples in Y</returns>
        public double[,] ValidateUsingRMSwithPredictedY(double[,] Ypredict, double[,] Ytest)
        {
            Subtract(ref Ypredict, Ytest);         // absolute difference between PREDICT and REAL
            return MeanRowWise(Square(Ypredict));  // mean(difference ^ 2)
        }


        // ====================================== PRIVATE ======================================


        private bool CheckInput(double[,] X, double[,] Y)
        {
            bool isSastifiedInput = false;
            //int nRowsOfX = X.GetUpperBound(0) + 1;      int nRowsOfY = Y.GetUpperBound(0) + 1;
            //int nColsOfX = X.GetUpperBound(1) + 1;      int nColsOfY = Y.GetUpperBound(1) + 1;            

            if (X.GetUpperBound(0) + 1 > 0)
                if (X.GetUpperBound(1) + 1 > 0)
                    if (X.GetUpperBound(0) + 1 == Y.GetUpperBound(0) + 1)
                        if (Y.GetUpperBound(1) + 1 > 0) isSastifiedInput = true;
                        else throw new Exception("The input data Y must as least one variable.");
                    else throw new Exception("The input data Y must have the same number of samples as of X.");
                else throw new Exception("The input data X must have at least one feature.");
            else throw new Exception("The input data X must have at least one sample.");
            return isSastifiedInput;
        }


        private void DeflateXY(int iComponent, ref double[,] V, ref double[,] XY)
        {
            // initialize a vector for deflate
            double[,] v0 = new double[this.nFeatures, 1];
            CopyOneColumn(this.P, iComponent, ref v0, 0); // initialize orthogonal loadings                     // v0 = P(:, i)

            // deflate v0
            for (int repeat = 0; repeat < 2; repeat++)
            {
                for (int j = 0; j < iComponent; j++) // is not accessible at the first component
                {
                    double[,] vj = new double[this.nFeatures, 1];
                    CopyOneColumn(V, j, ref vj, 0);                                                             // vj = V(:, col)

                    // v0 = v0 - (v0' * vj) * vj
                    double[,] tmp1 = CrossProduct(Transpose(v0), vj);                                           // v0' * vj
                    double[,] tmp2 = Multiply(vj, tmp1[0, 0]);                                                  // vj * (v0' * vj)
                    Subtract(ref v0, tmp2);                                                                     // v0 = v0 - (v0' * vj) * vj
                }
            }

            // normalize v0
            double v0norm = Norm(v0);
            v0 = Divide(v0, v0norm);                                                                            // v0 = v0 / ||v0||
            CopyOneColumn(v0, 0, ref V, iComponent);                                                            // V(:, i) = v0;

            // deflate: XY = XY - v0 * (v0' * XY)
            double[,] tmp3 = CrossProduct(Transpose(v0), XY);                                                   // v0' * XY
            double[,] tmp4;
            if (this.nDimY == 1) tmp4 = Multiply(v0, tmp3[0, 0]);                                               // v0 * (v0' * XY)
            else tmp4 = CrossProduct(v0, tmp3);
            Subtract(ref XY, tmp4);                                                                             // XY = XY - v0 * (v0' * XY)

            // deflate: XY = XY - Vi * (Vi' * XY)
            double[,] Vi = new double[this.nFeatures, iComponent + 1];
            CopyColumns(V, 0, iComponent, ref Vi, 0);                                                           // Vi = V(:, 0:i)

            double[,] tmp5 = CrossProduct(Transpose(Vi), XY);                                                   // Vi' * XY
            double[,] tmp6;
            if (this.nDimY == 1) tmp6 = Multiply(Vi, tmp5[0, 0]);                                               // Vi * (Vi' * XY)
            else tmp6 = CrossProduct(Vi, tmp5);
            Subtract(ref XY, tmp6);                                                                             // XY = XY - Vi * (Vi' * XY)
        }


        private void CalculateB_regressionMatrix()
        {            
            double[,] WQt           = CrossProduct(this.W, Transpose(this.Q));// nFeatures x nDimY                            // W * Q'

            // calculate meanOfX and meanOfY
            double[,] meanOfX       = MeanColumnWise(X);                      // 1 x nFeatures                                // mean(x, 1)            
            double[,] meanOfXAndWQt = CrossProduct(meanOfX, WQt);             // 1 x nDimY                                    // meanOfX * WQt
            double[,] meanOfY       = MeanColumnWise(Y);                      // 1 x nDimY                                    // mean(y, 1)            

            // calculate the intercept of X and Y
            Subtract(ref meanOfY, meanOfXAndWQt);                             // 1 x nDimY                                    // meanOfY = meanOfY - meanOfXAndWQt
            this.B0                 = meanOfY;                                // 1 x nDimY                                    // intercept B0 = meanOfY

            // copy the calculated values from B0 and WQt to B
            CopyOneRow(this.B0, 0, ref this.B, 0);
            CopyRows(WQt, 0, this.nFeatures - 1, ref this.B, 1);
        }


        private void DeflateU(int nComponents)
        {
            for (int iComponent = 0; iComponent < nComponents; iComponent++)
            {
                // copy u0 as a vector of U
                double[,] u0 = new double[this.nSamples, 1];
                CopyOneColumn(this.U, iComponent, ref u0, 0);               // u0 = U(:, i)

                // deflate
                for (int repeat = 0; repeat < 2; repeat++)
                {
                    for (int j = 0; j < iComponent; j++)
                    {
                        double[,] tj = new double[this.nSamples, 1];
                        CopyOneColumn(this.T, j, ref tj, 0);                // tj = T(:, col)

                        // u0 = u0 - (u0' * tj) * tj
                        double[,] tmp1 = CrossProduct(Transpose(u0), tj);   // u0' * tj
                        double[,] tmp2 = Multiply(tj, tmp1[0, 0]);          // (u0' * tj) * tj
                        Subtract(ref u0, tmp2);                             // u0 = u0 - (u0' * tj) * tj
                    }
                }

                // assign u0 back to U
                CopyOneColumn(u0, 0, ref this.U, iComponent);
            }
        }


        // ------------- for matrix copy -------------


        private void CopyOneColumn(double[,] input, int inputColumnIdx, ref double[,] output, int outputColumnIdx)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;          int nRowsOfOutput = output.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;          int nColsOfOutput = output.GetUpperBound(1) + 1;

            if (nRowsOfInput > nRowsOfOutput)        throw new Exception("Number of rows of the ouput must be larger than of the input.");
            if (outputColumnIdx > nColsOfOutput - 1) throw new Exception("Column index of the output is out of range.");
            
            for (int i = 0; i < nRowsOfInput; i++)
                output[i, outputColumnIdx] = input[i, inputColumnIdx];
        }


        private void CopyColumns(double[,] input,      int inputColumnIdxFrom,  int inputColumnIdxTo,
                                 ref double[,] output, int outputColumnIdxFrom)//, int outputColumnIdxTo)
        {
            /* // Method 1: Straightforward
            int nRowsOfInput = input.GetUpperBound(0) + 1;      int nRowsOfOutput = output.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;      int nColsOfOutput = output.GetUpperBound(1) + 1;

            if (nRowsOfInput > nRowsOfOutput)
                throw new Exception("Number of rows of the ouput must be larger than of the input.");
            if (inputColumnIdxTo - inputColumnIdxFrom > nColsOfOutput - outputColumnIdxFrom)
                throw new Exception("Column indices are incorrect.");

            for (int i = 0; i < nRowsOfInput; i++)
                for (int jIn = inputColumnIdxFrom, jOut = outputColumnIdxFrom; jIn <= inputColumnIdxTo; jIn++, jOut++)
                    output[i, jOut] = input[i, jIn];
            */

            // Method 2: Utilize CopyOneColumn()
            for (int jIn = inputColumnIdxFrom, jOut = outputColumnIdxFrom; jIn <= inputColumnIdxTo; jIn++, jOut++)
                this.CopyOneColumn(input, jIn, ref output, jOut);
        }


        private void CopyOneRow(double[,] input, int inputRowIdx, ref double[,] output, int outputRowIdx)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;      int nRowsOfOutput = output.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;      int nColsOfOutput = output.GetUpperBound(1) + 1;

            if (nColsOfInput > nColsOfOutput)     throw new Exception("Number of columns of the ouput must be larger than of the input.");
            if (outputRowIdx > nRowsOfOutput - 1) throw new Exception("Row index of the output is out of range.");

            for (int i = 0; i < nColsOfInput; i++)
                output[outputRowIdx, i] = input[inputRowIdx, i];
        }


        private void CopyRows(double[,] input,      int inputRowIdxFrom, int inputRowIdxTo,
                              ref double[,] output, int outputRowIdxFrom)
        {
            for (int jIn = inputRowIdxFrom, jOut = outputRowIdxFrom; jIn <= inputRowIdxTo; jIn++, jOut++)
                this.CopyOneRow(input, jIn, ref output, jOut);
        }


        // ------------- for matrix calculation -------------
        
        private double[,] Transpose(double[,] input)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;

            double[,] output = new double[nColsOfInput, nRowsOfInput];
            for (int i = 0; i < nRowsOfInput; i++)
                for (int j = 0; j < nColsOfInput; j++)
                    output[j, i] = input[i, j];
            return output;
        }


        private double Norm(double[,] input)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;

            double output = 0;
            for (int i = 0; i < nRowsOfInput; i++)
                for (int j = 0; j < nColsOfInput; j++)
                    output += input[i, j] * input[i, j];
            return Math.Sqrt(output);
        }


        private double Sum(double[,] input)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;

            double output = 0;
            for (int j = 0; j < nColsOfInput; j++)
                for (int i = 0; i < nRowsOfInput; i++)
                    output += input[i, j];
            return output;
        }


        private double[,] Square(double[,] input)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;

            double[,] output = new double[nRowsOfInput, nColsOfInput];
            for (int i = 0; i < nRowsOfInput; i++)
                for (int j = 0; j < nColsOfInput; j++)
                    output[i, j] = input[i, j] * input[i, j];
            return output;
        }


        private void Subtract(ref double[,] input1, double[,] input2)
        {
            int nRowsOfInput = input1.GetUpperBound(0) + 1;
            if (nRowsOfInput != input2.GetUpperBound(0) + 1) throw new Exception("Input1 nRows != Input2 nRows.");

            int nColsOfInput = input1.GetUpperBound(1) + 1;
            if (nColsOfInput != input2.GetUpperBound(1) + 1) throw new Exception("Input1 nCols != Input2 nCols.");

            for (int i = 0; i < nRowsOfInput; i++)
                for (int j = 0; j < nColsOfInput; j++)
                    input1[i, j] = input1[i, j] - input2[i, j];
        }


        private double[,] Multiply(double[,] input, double number)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;

            double[,] output = new double[nRowsOfInput, nColsOfInput];
            for (int i = 0; i < nRowsOfInput; i++)
                for (int j = 0; j < nColsOfInput; j++)
                    output[i, j] = input[i, j] * number;
            return output;
        }


        private double[,] Divide(double[,] input, double number)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;

            double[,] output = new double[nRowsOfInput, nColsOfInput];
            for (int i = 0; i < nRowsOfInput; i++)
                for (int j = 0; j < nColsOfInput; j++)
                    output[i, j] = input[i, j] / number;
            return output;
        }


        private double[,] CrossProduct(double[,] input1, double[,] input2)
        {
            // input1: n x m
            int n = input1.GetUpperBound(0) + 1;
            int m = input1.GetUpperBound(1) + 1;

            // input2: m x k
            int k = input2.GetUpperBound(1) + 1;

            if (m != input2.GetUpperBound(0) + 1)
                throw new Exception("Input1 nCols != Input2 nRows.");

            // multiply
            double[,] output = new double[n, k];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < k; j++)
                    for (int ij = 0; ij < m; ij++)
                        output[i, j] += input1[i, ij] * input2[ij, j];
            return output;
        }


        private void SVD(double[,] input, out double[,] S, out double[] V, out double[,] Dt)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;
            S = new double[0, 0];
            V = new double[0];
            Dt = new double[0, 0];
            alglib.svd.rmatrixsvd(input, nRowsOfInput, nColsOfInput, 1, 1, 2, ref V, ref S, ref Dt);
        }


        private double[,] SumColumnWise(double[,] input)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;

            double[,] output = new double[1, nColsOfInput];
            for (int j = 0; j < nColsOfInput; j++)
                for (int i = 0; i < nRowsOfInput; i++)
                    output[0, j] += input[i, j];
            return output;
        }


        private double[,] MeanColumnWise(double[,] input)
        {            
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;

            double[,] output = new double[1, nColsOfInput];            
            for (int j = 0; j < nColsOfInput; j++)
            {
                double sum = 0;
                for (int i = 0; i < nRowsOfInput; i++)
                     sum += input[i, j];
                output[0, j] = sum / nRowsOfInput;
            }
            return output;
        }


        private double[,] MeanRowWise(double[,] input)
        {
            int nRowsOfInput = input.GetUpperBound(0) + 1;
            int nColsOfInput = input.GetUpperBound(1) + 1;

            double[,] output = new double[nRowsOfInput, 1];
            for (int i = 0; i < nRowsOfInput; i++)
            {
                double sum = 0;
                for (int j = 0; j < nColsOfInput; j++)
                    sum += input[i, j];
                output[i, 0] = sum / nColsOfInput;
            }
            return output;
        }
    }
}
