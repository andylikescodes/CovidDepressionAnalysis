w6_data = read.csv('./output/Tetrad_output/w6.csv')
wilcox.test(Depression ~ Mandatory_SAH, data=w6_data) 
t.test(Depression ~ Mandatory_SAH,, var.equal=TRUE, data = w6_data)

w8_data = read.csv('./output/Tetrad_output/w8.csv')
wilcox.test(Depression ~ Mandatory_SAH, data=w8_data) 
t.test(Depression ~ Mandatory_SAH,, var.equal=TRUE, data = w8_data)

w10_data = read.csv('./output/Tetrad_output/w10.csv')
wilcox.test(Depression ~ Mandatory_SAH, data=w10_data) 
t.test(Depression ~ Mandatory_SAH,, var.equal=TRUE, data = w10_data)

w12_data = read.csv('./output/Tetrad_output/w12.csv')
wilcox.test(Depression ~ Mandatory_SAH, data=w12_data) 
t.test(Depression ~ Mandatory_SAH,, var.equal=TRUE, data = w12_data)

w14_data = read.csv('./output/Tetrad_output/w14.csv')
wilcox.test(Depression ~ Mandatory_SAH, data=w14_data) 
t.test(Depression ~ Mandatory_SAH,, var.equal=TRUE, data = w14_data)

w16_data = read.csv('./output/Tetrad_output/w16.csv')
wilcox.test(Depression ~ Mandatory_SAH, data=w16_data) 
t.test(Depression ~ Mandatory_SAH,, var.equal=TRUE, data = w16_data)

