# Violent Pose Detection

This project implements an Artificial Neural Network aimed at detecting "violent" poses, essentially it detects hands up with "erratic" posture.

It works by using a neural network to process the poses for an individual photo, afterwards a fully connected neural network will be trained with the poses and the labels (violent pose vs nonviolent pose).

This was done during a master's degree class on Neural Networks, so don't expect production-grade training and debuggability, it was an interesting project but it can be extended in many ways.

## Improvements

I believe this can be improved by not simply using poses to detect threats and using higher definition cameras since those are ubiquitous at the moment.
First, there are detectable expressions on the human face when one is tense, and also the gun itself can be detected using an object detection phase.



