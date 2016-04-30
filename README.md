# AIProject3

Install/Run guide

1. Clone this repository to your computer
2. Install Anaconda from https://www.continuum.io/downloads, for numpy, scipy, and scikit-learn
3. Run 'conda install Pillow scipy numpy scikit-learn' in the terminal to make sure they're installed and updated
4. Run program using 'ipython classify.py <insert image path here>'
5. If the classify program is outputting an error, or has low accuracy, it probably means the saved decision trees are from a different version of scikit-learn than you are running. To fix this, run 'ipython training.py'. This will train the decision trees again on the sample data.
