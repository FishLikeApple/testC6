# testC6
This is a test for learning Transfer Learning with the model of inception_v3.

The photo folder “flower_photos” (hidden) is a folder of test data which need to be downloaded from Google, and the checkpoint file inception_v3.ckpt should also be downloaded from Google.

The result just not good with a final accuracy rate below 0.3, so is there anybody who can help me in finding bugs causing this result?

# 2018/7/13：
I finally found out where the problem is. All the variates in the inception_v3.ckpt are defined under a scope, so that I should define the model under the scope too by using "with slim.arg_scope(inception_v3_arg_scope()):".

See:https://stackoverflow.com/questions/39357454/restore-checkpoint-in-tensorflow-tensor-name-not-found and the answer 2 in that.
