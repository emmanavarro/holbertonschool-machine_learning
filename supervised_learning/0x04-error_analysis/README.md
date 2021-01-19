# 0x04. Error Analysis

## Learning Objectives

* What is the confusion matrix?
* What is type I error? type II?
* What is sensitivity? specificity? precision? recall?
* What is an F1 score?
* What is bias? variance?
* What is irreducible error?
* What is Bayes error?
* How can you approximate Bayes error?
* How to calculate bias and variance
* How to create a confusion matrix

## Tasks

0. Create Confusion
   * Write the function `def create_confusion_matrix(labels, logits):` that creates a confusion matrix.
1. Sensitivity
   * Write the function `def sensitivity(confusion):` that calculates the sensitivity for each class in a confusion matrix
2. Precision
   * Write the function `def precision(confusion):` that calculates the precision for each class in a confusion matrix
3. Specificity
   * Write the function `def specificity(confusion):` that calculates the specificity for each class in a confusion matrix
4. F1 score
   * Write the function `def f1_score(confusion):` that calculates the F1 score of a confusion matrix
5. Dealing with Error
   * n the text file `5-error_handling`, write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex. `A,B,C`):

   * Scenarios:

   ```
    1. High Bias, High Variance
    2. High Bias, Low Variance
    3. Low Bias, High Variance
    4. Low Bias, Low Variance
    ```

   * Approaches:
    ```
    A. Train more
    B. Try a different architecture
    C. Get more data
    D. Build a deeper network
    E. Use regularization
    F. Nothing
   ```
6. Compare and Contrast
   * Given the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is and write the lettered answer in the file `6-compare_and_contrast`
   ![Confusion Training Matrix](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/03c511c109a790a30bbe.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210118%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210118T210323Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4e6a8dc31025356ad9ea5b3c610b17b9bc2f5aaf9e7ebf0c759f537087fa210f)
   ![Confusion Validation Matrix](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/8f5d5fdab6420a22471b.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210118%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210118T210323Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f65d266f73e4b989214a3d5d072faea10f98080c71fc7eb95878ee339f7cbbeb)
   Most important issue:
   ```
   A. High Bias
   B. High Variance
   C. Nothing
   ```
