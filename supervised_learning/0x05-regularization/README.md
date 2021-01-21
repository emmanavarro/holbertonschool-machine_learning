<h1 class="gap">0x05. Regularization</h1>

<article id="description" class="gap formatted-content">
    <p><a href="https://ibb.co/gwDkbFn"><img src="https://i.ibb.co/6b4jcJd/689c11afbc30eaa89b50.jpg" alt="689c11afbc30eaa89b50" border="0"></a></p>

<h2>Resources</h2>

<p><strong>Read or watch</strong>:</p>

<ul>
<li><a href="/rltoken/G22TZHYwwb0PwlAuEZdDEQ" title="Regularization (mathematics)" target="_blank">Regularization (mathematics)</a> </li>
<li><a href="/rltoken/Mao_NUBBiwm0Qh8b-axAgw" title="An Overview of Regularization Techniques in Deep Learning" target="_blank">An Overview of Regularization Techniques in Deep Learning</a> <strong>(up to <code>A case study on MNIST data with keras</code> excluded)</strong></li>
<li><a href="/rltoken/AY80ruaSMDL_AGnjZOpWGQ" title="L2 Regularization and Back-Propagation" target="_blank">L2 Regularization and Back-Propagation</a></li>
<li><a href="/rltoken/LANl57haitMwBGXpTq1q8Q" title="Intuitions on L1 and L2 Regularisation" target="_blank">Intuitions on L1 and L2 Regularisation</a></li>
<li><a href="/rltoken/huRNIkxWr5OV1Tit658LcQ" title="Analysis of Dropout" target="_blank">Analysis of Dropout</a> </li>
<li><a href="/rltoken/4YMCmw41ovvYtMvr-Wl7LA" title="Early stopping" target="_blank">Early stopping</a> </li>
<li><a href="/rltoken/t6UPkGJXD_nK7TfGwE9Rig" title="How to use early stopping properly for training deep neural network?" target="_blank">How to use early stopping properly for training deep neural network?</a> </li>
<li><a href="/rltoken/MaLMSTSCPux71mW1RIhiBA" title="Data Augmentation | How to use Deep Learning when you have Limited Data " target="_blank">Data Augmentation | How to use Deep Learning when you have Limited Data </a> </li>
<li><a href="/rltoken/GriJE79Gr4BF8HG2DGpbYg" title="deeplearning.ai" target="_blank">deeplearning.ai</a> videos (<em>Note: I suggest watching these video at 1.5x - 2x speed</em>):

<ul>
<li><a href="/rltoken/BJoxOnJN-GJyZ_fJ9qT0EQ" title="Regularization" target="_blank">Regularization</a></li>
<li><a href="/rltoken/dLdv5Gi77DmWNyR3MHe69g" title="Why Regularization Reduces Overfitting" target="_blank">Why Regularization Reduces Overfitting</a></li>
<li><a href="/rltoken/23ue4EQxNd9LOCW0Q6FNNQ" title="Dropout Regularization" target="_blank">Dropout Regularization</a></li>
<li><a href="/rltoken/eleB8ZvoJiOltULeHkDvGQ" title="Understanding Dropout" target="_blank">Understanding Dropout</a></li>
<li><a href="/rltoken/QuFgq0_MKTGq6UAKj5OjEw" title="Other Regularization Methods" target="_blank">Other Regularization Methods</a></li>
</ul></li>
</ul>

<p><strong>References</strong>:</p>

<ul>
<li><a href="/rltoken/TwoE2r0JTScpR1EivtkgGQ" title="numpy.linalg.norm" target="_blank">numpy.linalg.norm</a> </li>
<li><a href="/rltoken/Ffuh27LbhV1S-Am0oXgwlg" title="numpy.random.binomial" target="_blank">numpy.random.binomial</a> </li>
<li><a href="/rltoken/qYvQGggYVUx4qIc5Ft0Zqw" title="tf.contrib.layers.l2_regularizer" target="_blank">tf.contrib.layers.l2_regularizer</a> </li>
<li><a href="/rltoken/UFA3JMNu_cvqljoYjcNqcg" title="tf.layers.Dense#kernel_regularizer" target="_blank">tf.layers.Dense#kernel_regularizer</a> </li>
<li><a href="/rltoken/zZvrk-RUoJ1zNj43rmllQQ" title="tf.losses.get_regularization_loss" target="_blank">tf.losses.get_regularization_loss</a> </li>
<li><a href="/rltoken/eCkBE5CmSlTpSc0Sh6W_vg" title="tf.layers.Dropout" target="_blank">tf.layers.Dropout</a> </li>
<li><a href="/rltoken/2jIHjQpd_A2-4IF1SbL5dg" title="Dropout: A Simple Way to Prevent Neural Networks from Overfitting" target="_blank">Dropout: A Simple Way to Prevent Neural Networks from Overfitting</a> </li>
<li><a href="/rltoken/b_knZ8MqBEHA3TPoGruYGw" title="Early Stopping - but when?" target="_blank">Early Stopping - but when?</a> </li>
<li><a href="/rltoken/JVvKoC0p-wBoLl3qF7xChQ" title="L2 Regularization versus Batch and Weight Normalization" target="_blank">L2 Regularization versus Batch and Weight Normalization</a> </li>
</ul>

<h2>Learning Objectives</h2>

<p>At the end of this project, you are expected to be able to <a href="/rltoken/MWoxlOvUqjg7BCkjekSVfA" title="explain to anyone" target="_blank">explain to anyone</a>, <strong>without the help of Google</strong>:</p>

<h3>General</h3>

<ul>
<li>What is regularization? What is its purpose?</li>
<li>What is are L1 and L2 regularization? What is the difference between the two methods?</li>
<li>What is dropout?</li>
<li>What is early stopping?</li>
<li>What is data augmentation?</li>
<li>How do you implement the above regularization methods in Numpy? Tensorflow?</li>
<li>What are the pros and cons of the above regularization methods?</li>
</ul>

<h2>Requirements</h2>

<h3>General</h3>

<ul>
<li>Allowed editors: <code>vi</code>, <code>vim</code>, <code>emacs</code></li>
<li>All your files will be interpreted/compiled on Ubuntu 16.04 LTS using <code>python3</code> (version 3.5)</li>
<li>Your files will be executed with <code>numpy</code> (version 1.15)</li>
<li>All your files should end with a new line</li>
<li>The first line of all your files should be exactly <code>#!/usr/bin/env python3</code></li>
<li>A <code>README.md</code> file, at the root of the folder of the project, is mandatory</li>
<li>Your code should use the <code>pycodestyle</code> style (version 2.4)</li>
<li>All your modules should have documentation (<code>python3 -c 'print(__import__("my_module").__doc__)'</code>)</li>
<li>All your classes should have documentation (<code>python3 -c 'print(__import__("my_module").MyClass.__doc__)'</code>)</li>
<li>All your functions (inside and outside a class) should have documentation (<code>python3 -c 'print(__import__("my_module").my_function.__doc__)'</code> and <code>python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'</code>)</li>
<li>Unless otherwise noted, you are not allowed to import any module except <code>import numpy as np</code> and <code>import tensorflow as tf</code></li>
<li>You are not allowed to use the <code>keras</code> module in <code>tensorflow</code></li>
<li>You should not import any module unless it is being used</li>
<li>All your files must be executable</li>
<li>The length of your files will be tested using <code>wc</code></li>
<li>When initializing layer weights, use <code>tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")</code>.</li>
</ul>

  </article>
