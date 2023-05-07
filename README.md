Download Link: https://assignmentchef.com/product/solved-machinelearning-exercise-5-regularized-linear-regression-and-bias-v-s-variance
<br>
<h1>Introduction</h1>

In this exercise, you will implement regularized linear regression and use it to study models with different bias-variance properties. Before starting on the programming exercise, we strongly recommend watching the video lectures and completing the review questions for the associated topics.

To get started with the exercise, you will need to download the starter code and unzip its contents to the directory where you wish to complete the exercise. If needed, use the cd command in Octave/MATLAB to change to this directory before starting this exercise.

You can also find instructions for installing Octave/MATLAB in the “Environment Setup Instructions” of the course website.

<h2>Files included in this exercise</h2>

ex5.m – Octave/MATLAB script that steps you through the exercise ex5data1.mat – Dataset

submit.m – Submission script that sends your solutions to our servers featureNormalize.m – Feature normalization function fmincg.m – Function minimization routine (similar to fminunc) plotFit.m – Plot a polynomial fit trainLinearReg.m – Trains linear regression using your cost function [<em>?</em>] linearRegCostFunction.m – Regularized linear regression cost function

[<em>?</em>] learningCurve.m – Generates a learning curve

[<em>?</em>] polyFeatures.m – Maps data into polynomial feature space

[<em>?</em>] validationCurve.m – Generates a cross validation curve

<em>? </em>indicates files you will need to complete

Throughout the exercise, you will be using the script ex5.m. These scripts set up the dataset for the problems and make calls to functions that you will write. You are only required to modify functions in other files, by following the instructions in this assignment.

<h2>Where to get help</h2>

The exercises in this course use Octave<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a> or MATLAB, a high-level programming language well-suited for numerical computations. If you do not have Octave or MATLAB installed, please refer to the installation instructions in the “Environment Setup Instructions” of the course website.

At the Octave/MATLAB command line, typing help followed by a function name displays documentation for a built-in function. For example, help plot will bring up help information for plotting. Further documentation for Octave functions can be found at the <a href="https://www.gnu.org/software/octave/doc/interpreter/">Octave documentation pages</a><a href="https://www.gnu.org/software/octave/doc/interpreter/">.</a> MATLAB documentation can be found at the <a href="https://www.mathworks.com/help/matlab/?refresh=true">MATLAB documentation pages</a><a href="https://www.mathworks.com/help/matlab/?refresh=true">.</a>

We also strongly encourage using the online <strong>Discussions </strong>to discuss exercises with other students. However, do not look at any source code written by others or share your source code with others.

<h1>1          Regularized Linear Regression</h1>

In the first half of the exercise, you will implement regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir. In the next half, you will go through some diagnostics of debugging learning algorithms and examine the effects of bias v.s. variance.

The provided script, ex5.m, will help you step through this exercise.

<h2>1.1        Visualizing the dataset</h2>

We will begin by visualizing the dataset containing historical records on the change in the water level, <em>x</em>, and the amount of water flowing out of the dam, <em>y</em>.

This dataset is divided into three parts:

<ul>

 <li>A <strong>training </strong>set that your model will learn on: X, y</li>

 <li>A <strong>cross validation </strong>set for determining the regularization parameter:</li>

</ul>

Xval, yval

<ul>

 <li>A <strong>test </strong>set for evaluating performance. These are “unseen” examples which your model did not see during training: Xtest, ytest</li>

</ul>

The next step of ex5.m will plot the training data (Figure 1). In the following parts, you will implement linear regression and use that to fit a straight line to the data and plot learning curves. Following that, you will implement polynomial regression to find a better fit to the data.

Figure 1: Data

<h2>1.2        Regularized linear regression cost function</h2>

Recall that regularized linear regression has the following cost function:

<em> ,</em>

where <em>λ </em>is a regularization parameter which controls the degree of regularization (thus, help preventing overfitting). The regularization term puts a penalty on the overal cost <em>J</em>. As the magnitudes of the model parameters <em>θ<sub>j </sub></em>increase, the penalty increases as well. Note that you should not regularize the <em>θ</em><sub>0 </sub>term. (In Octave/MATLAB, the <em>θ</em><sub>0 </sub>term is represented as theta(1) since indexing in Octave/MATLAB starts from 1).

You should now complete the code in the file linearRegCostFunction.m. Your task is to write a function to calculate the regularized linear regression cost function. If possible, try to vectorize your code and avoid writing loops. When you are finished, the next part of ex5.m will run your cost function using theta initialized at [1; 1]. You should expect to see an output of

303.993.

<em>You should now submit your solutions.</em>

<h2>1.3        Regularized linear regression gradient</h2>

Correspondingly, the partial derivative of regularized linear regression’s cost for <em>θ<sub>j </sub></em>is defined as

for <em>j </em>= 0

for <em>j </em>≥ 1

In linearRegCostFunction.m, add code to calculate the gradient, returning it in the variable grad. When you are finished, the next part of ex5.m will run your gradient function using theta initialized at [1; 1]. You should expect to see a gradient of [-15.30; 598.250].

<em>You should now submit your solutions.</em>

<h2>1.4        Fitting linear regression</h2>

Once your cost function and gradient are working correctly, the next part of ex5.m will run the code in trainLinearReg.m to compute the optimal values of <em>θ</em>. This training function uses fmincg to optimize the cost function.

In this part, we set regularization parameter <em>λ </em>to zero. Because our current implementation of linear regression is trying to fit a 2-dimensional <em>θ</em>, regularization will not be incredibly helpful for a <em>θ </em>of such low dimension. In the later parts of the exercise, you will be using polynomial regression with regularization.

Finally, the ex5.m script should also plot the best fit line, resulting in an image similar to Figure 2. The best fit line tells us that the model is not a good fit to the data because the data has a non-linear pattern. While visualizing the best fit as shown is one possible way to debug your learning algorithm, it is not always easy to visualize the data and model. In the next section, you will implement a function to generate learning curves that can help you debug your learning algorithm even if it is not easy to visualize the data.

Figure 2: Linear Fit

<h1>2          Bias-variance</h1>

An important concept in machine learning is the bias-variance tradeoff. Models with high bias are not complex enough for the data and tend to underfit, while models with high variance overfit to the training data.

In this part of the exercise, you will plot training and test errors on a learning curve to diagnose bias-variance problems.

<h2>2.1        Learning curves</h2>

You will now implement code to generate the learning curves that will be useful in debugging learning algorithms. Recall that a learning curve plots training and cross validation error as a function of training set size. Your job is to fill in learningCurve.m so that it returns a vector of errors for the training set and cross validation set.

To plot the learning curve, we need a training and cross validation set error for different <em>training </em>set sizes. To obtain different training set sizes, you should use different subsets of the original training set X. Specifically, for a training set size of i, you should use the first i examples (i.e., X(1:i,:) and y(1:i)).

You can use the trainLinearReg function to find the <em>θ </em>parameters. Note that the lambda is passed as a parameter to the learningCurve function. After learning the <em>θ </em>parameters, you should compute the <strong>error </strong>on the training and cross validation sets. Recall that the training error for a dataset is defined as

<em> .</em>

In particular, note that the training error does not include the regularization term. One way to compute the training error is to use your existing cost function and set <em>λ </em>to 0 <em>only </em>when using it to compute the training error and cross validation error. When you are computing the training set error, make sure you compute it on the training subset (i.e., X(1:n,:) and y(1:n)) (instead of the entire training set). However, for the cross validation error, you should compute it over the <em>entire </em>cross validation set. You should store the computed errors in the vectors error train and error val.

When you are finished, ex5.m wil print the learning curves and produce a plot similar to Figure 3.

<em>You should now submit your solutions.</em>

In Figure 3, you can observe that <em>both </em>the train error and cross validation error are high when the number of training examples is increased. This reflects a <strong>high bias </strong>problem in the model – the linear regression model is

Learning curve for linear regression

Figure 3: Linear regression learning curve

too simple and is unable to fit our dataset well. In the next section, you will implement polynomial regression to fit a better model for this dataset.

<h1>3          Polynomial regression</h1>

The problem with our linear model was that it was too simple for the data and resulted in underfitting (high bias). In this part of the exercise, you will address this problem by adding more features.

For use polynomial regression, our hypothesis has the form:

<em>h<sub>θ</sub></em>(<em>x</em>) = <em>θ</em><sub>0 </sub>+ <em>θ</em><sub>1 </sub>∗ (waterLevel) + <em>θ</em><sub>2 </sub>∗ (waterLevel)<sup>2 </sup>+ ··· + <em>θ<sub>p </sub></em>∗ (waterLevel)<em><sup>p </sup></em>= <em>θ</em><sub>0 </sub>+ <em>θ</em><sub>1</sub><em>x</em><sub>1 </sub>+ <em>θ</em><sub>2</sub><em>x</em><sub>2 </sub>+ <em>… </em>+ <em>θ<sub>p</sub>x<sub>p</sub>.</em>

Notice that by defining <em>x</em><sub>1 </sub>= (waterLevel)<em>,x</em><sub>2 </sub>= (waterLevel)<sup>2</sup><em>,…,x<sub>p </sub></em>= (waterLevel)<em><sup>p</sup></em>, we obtain a linear regression model where the features are the various powers of the original value (waterLevel).

Now, you will add more features using the higher powers of the existing feature <em>x </em>in the dataset. Your task in this part is to complete the code in polyFeatures.m so that the function maps the original training set X of size <em>m</em>×1 into its higher powers. Specifically, when a training set X of size <em>m</em>×1 is passed into the function, the function should return a <em>m</em>×<em>p </em>matrix X poly, where column 1 holds the original values of X, column 2 holds the values of X.^2, column 3 holds the values of X.^3, and so on. Note that you don’t have to account for the zero-eth power in this function.

Now you have a function that will map features to a higher dimension, and Part 6 of ex5.m will apply it to the training set, the test set, and the cross validation set (which you haven’t used yet).

<em>You should now submit your solutions.</em>

<h2>3.1        Learning Polynomial Regression</h2>

After you have completed polyFeatures.m, the ex5.m script will proceed to train polynomial regression using your linear regression cost function.

Keep in mind that even though we have polynomial terms in our feature vector, we are still solving a linear regression optimization problem. The polynomial terms have simply turned into features that we can use for linear regression. We are using the same cost function and gradient that you wrote for the earlier part of this exercise.

For this part of the exercise, you will be using a polynomial of degree 8. It turns out that if we run the training directly on the projected data, will not work well as the features would be badly scaled (e.g., an example with <em>x </em>= 40 will now have a feature <em>x</em><sub>8 </sub>= 40<sup>8 </sup>= 6<em>.</em>5 × 10<sup>12</sup>). Therefore, you will need to use feature normalization.

Before learning the parameters <em>θ </em>for the polynomial regression, ex5.m will first call featureNormalize and normalize the features of the training set, storing the mu, sigma parameters separately. We have already implemented this function for you and it is the same function from the first exercise.

After learning the parameters <em>θ</em>, you should see two plots (Figure 4,5) generated for polynomial regression with <em>λ </em>= 0.

From Figure 4, you should see that the polynomial fit is able to follow the datapoints very well – thus, obtaining a low training error. However, the polynomial fit is very complex and even drops off at the extremes. This is an indicator that the polynomial regression model is overfitting the training data and will not generalize well.

To better understand the problems with the unregularized (<em>λ </em>= 0) model, you can see that the learning curve (Figure 5) shows the same effect where the low training error is low, but the cross validation error is high. There is a gap between the training and cross validation errors, indicating a high variance problem.

Figure 4: Polynomial fit, <em>λ </em>= 0

Polynomial Regression Learning Curve (lambda = 0.000000)

Figure 5: Polynomial learning curve, <em>λ </em>= 0

One way to combat the overfitting (high-variance) problem is to add regularization to the model. In the next section, you will get to try different <em>λ </em>parameters to see how regularization can lead to a better model.

<h2>3.2        Optional (ungraded) exercise: Adjusting the regularization parameter</h2>

In this section, you will get to observe how the regularization parameter affects the bias-variance of regularized polynomial regression. You should now modify the the lambda parameter in the ex5.m and try <em>λ </em>= 1<em>,</em>100. For each of these values, the script should generate a polynomial fit to the data and also a learning curve.

For <em>λ </em>= 1, you should see a polynomial fit that follows the data trend well (Figure 6) and a learning curve (Figure 7) showing that both the cross validation and training error converge to a relatively low value. This shows the <em>λ </em>= 1 regularized polynomial regression model does not have the highbias or high-variance problems. In effect, it achieves a good trade-off between bias and variance.

For <em>λ </em>= 100, you should see a polynomial fit (Figure 8) that does not follow the data well. In this case, there is too much regularization and the model is unable to fit the training data.

<em>You do not need to submit any solutions for this optional (ungraded) exercise.</em>

Figure 6: Polynomial fit, <em>λ </em>= 1

Polynomial Regression Learning Curve (lambda = 1.000000)

Figure 7: Polynomial learning curve, <em>λ </em>= 1

Figure 8: Polynomial fit, <em>λ </em>= 100

<h2>3.3        Selecting <em>λ </em>using a cross validation set</h2>

From the previous parts of the exercise, you observed that the value of <em>λ </em>can significantly affect the results of regularized polynomial regression on the training and cross validation set. In particular, a model without regularization (<em>λ </em>= 0) fits the training set well, but does not generalize. Conversely, a model with too much regularization (<em>λ </em>= 100) does not fit the training set and testing set well. A good choice of <em>λ </em>(e.g., <em>λ </em>= 1) can provide a good fit to the data.

In this section, you will implement an automated method to select the <em>λ </em>parameter. Concretely, you will use a cross validation set to evaluate how good each <em>λ </em>value is. After selecting the best <em>λ </em>value using the cross validation set, we can then evaluate the model on the test set to estimate how well the model will perform on actual unseen data.

Your task is to complete the code in validationCurve.m. Specifically, you should should use the trainLinearReg function to train the model using different values of <em>λ </em>and compute the training error and cross validation error.

You should try <em>λ </em>in the following range: {0<em>,</em>0<em>.</em>001<em>,</em>0<em>.</em>003<em>,</em>0<em>.</em>01<em>,</em>0<em>.</em>03<em>,</em>0<em>.</em>1<em>,</em>0<em>.</em>3<em>,</em>1<em>,</em>3<em>,</em>10}.

Figure 9: Selecting <em>λ </em>using a cross validation set

After you have completed the code, the next part of ex5.m will run your function can plot a cross validation curve of error v.s. <em>λ </em>that allows you select which <em>λ </em>parameter to use. You should see a plot similar to Figure 9. In this figure, we can see that the best value of <em>λ </em>is around 3. Due to randomness in the training and validation splits of the dataset, the cross validation error can sometimes be lower than the training error.

<em>You should now submit your solutions.</em>

<h2>3.4        Optional (ungraded) exercise: Computing test set error</h2>

In the previous part of the exercise, you implemented code to compute the cross validation error for various values of the regularization parameter <em>λ</em>. However, to get a better indication of the model’s performance in the real world, it is important to evaluate the “final” model on a test set that was not used in any part of training (that is, it was neither used to select the <em>λ </em>parameters, nor to learn the model parameters <em>θ</em>).

For this optional (ungraded) exercise, you should compute the test error using the best value of <em>λ </em>you found. In our cross validation, we obtained a test error of 3.8599 for <em>λ </em>= 3.

<em>You do not need to submit any solutions for this optional (ungraded) exercise.</em>

<h2>3.5        Optional (ungraded) exercise: Plotting learning curves with randomly selected examples</h2>

In practice, especially for small training sets, when you plot learning curves to debug your algorithms, it is often helpful to average across multiple sets of randomly selected examples to determine the training error and cross validation error.

Concretely, to determine the training error and cross validation error for <em>i </em>examples, you should first randomly select <em>i </em>examples from the training set and <em>i </em>examples from the cross validation set. You will then learn the parameters <em>θ </em>using the randomly chosen training set and evaluate the parameters <em>θ </em>on the randomly chosen training set and cross validation set. The above steps should then be repeated multiple times (say 50) and the averaged error should be used to determine the training error and cross validation error for <em>i </em>examples.

For this optional (ungraded) exercise, you should implement the above strategy for computing the learning curves. For reference, figure 10 shows the learning curve we obtained for polynomial regression with <em>λ </em>= 0<em>.</em>01. Your figure may differ slightly due to the random selection of examples.

<em>You do not need to submit any solutions for this optional (ungraded) exercise.</em>

Polynomial Regression Learning Curve (lambda = 0.010000)

Figure 10: Optional (ungraded) exercise: Learning curve with randomly selected examples

You are allowed to submit your solutions multiple times, and we will take only the highest score into consideration.

<a href="#_ftnref1" name="_ftn1">[1]</a> Octave is a free alternative to MATLAB. For the programming exercises, you are free to use either Octave or MATLAB.<img decoding="async" data-recalc-dims="1" data-src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2022/04/491.png?w=980&amp;ssl=1" class="lazyload" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

 <noscript>

  <img decoding="async" src="https://i0.wp.com/www.ankitcodinghub.com/wp-content/uploads/2022/04/491.png?w=980&amp;ssl=1" data-recalc-dims="1">

 </noscript>