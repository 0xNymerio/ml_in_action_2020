import tree
import tree_plotter
import numpy


# ====
figsize = (10,7)
# ===

fr = open('Decision_Tree/lenses.txt')
lenses_data = [inst.strip().split('\t') for inst in fr.readlines()]
lenses_label = ['age', 'prescript', 'astigmatic', 'tearRate']
lenses_tree = tree.createTree(lenses_data, lenses_label)

tree.storeTree(lenses_tree,'Decision_Tree/lenses_classifier.txt')


tree_plotter.createPlot(lenses_tree,figsize)

#print(tree.classify(dict_mytree,labels,[1,1]))
