from __future__ import print_function

from AI_Data.dataSource import DataSource

data_source = DataSource()

print("--------------------------多支A股日终行情数据-----------------------------------------")

wind_codes = ['600030.SH', '513100.SH']
start_date = '2017-12-01'
end_date = '2017-12-06'

# 测试A股港股多个万德代码日终行情,返回list
returns = data_source.getEODData(wind_codes, start_date, end_date)
print('\n')
print('*****************************%(wind_code)s日终行情***********************************' % {'wind_code': wind_codes})
print(returns)

# 必传时间字段,部分万得代码没有AVERAGE字段,传AVERAGE会报错
fields = ['DATE', 'WIND_CODE', 'CHANGE', 'OPEN', 'CLOSE']
returns = data_source.getEODData(wind_codes, start_date, end_date, fields=fields)
print('\n')
print('***************传递参数fields**%(wind_code)s日终行情*************' % {'wind_code': wind_codes})
print(returns)

print("\n")
