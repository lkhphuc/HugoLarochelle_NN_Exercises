from crf import LinearChainCRF

print "Verifying gradients without regularization"
m = LinearChainCRF()
m.L2 = 0
m.L1 = 0
m.verify_gradients()

print ""
print "Verifying gradients with L2 regularization"
m.L2 = 0.001
m.verify_gradients()

print ""
print "Verifying gradients with L1 regularization"
m.L2 = 0
m.L1 = 0.001
m.verify_gradients()

