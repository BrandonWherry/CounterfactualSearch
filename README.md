## __CSE__
Causal Subobject Explanations

For IJCAI 2023 submission

## __Abstract__

Automatic obfuscation of regions in images using
machine learning explanation has emerged as a crit-
ical problem in recent times. However, explanation
for image obfuscation demands pinpointing spe-
cific subobject regions of the input that are causal
of a model’s particular decision, because subobject
regions are human-understandable features, and
causal explanation enables a user to understand
what region caused a prediction for which it was
masked. Existing explanation approaches can
only provide attributions at a feature level and
cannot address the need for region-level causal
explanations. In this work, we propose a technique
called Causal Subobject Explanations (CSE) that
are counterfactual explanations produced based
on an adaptive region binary masking algorithm
with region attribution score heuristics, to identify
the regions in the image that caused the model’s
prediction to change. Extensive experiments on
a baseline dataset demonstrates the effectiveness
of CSE compared with five state-of-the-art ex-
plainability and four clustering approaches in
terms of three evaluation metrics. Furthermore,
we demonstrate the practicality of CSEs for three
datasets of harmful images, to automatically
obfuscate the harmful regions in these images,
thereby rendering them safe.
