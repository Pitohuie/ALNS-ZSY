The instances are formatted as follows:

###For each location the instance provides:
-StringId as a unique identifier
-Type indicates the function of the location, i.e,
---d: depot
---f: recharging station
---c: customer location
-x, y are coordinates (distances are assumed to be euclidean) 
-demand specifies the quantity of freight capacity required
-ReadyTime and DueDate are the beginning and the end of the time window (waiting is allowed)
-ServiceTime denotes the entire time spend at customer for loading operations

###For the electric vehicles (all identical):
-"Q Vehicle fuel tank capacity": units of energy available
-"C Vehicle load capacity":      units available for cargo
-"r fuel consumption rate":      reduction of battery capacity when traveling one unit of distance
-"g inverse refueling rate":     units of time required to recharge one unit of energy
-"v average Velocity":           assumed to be constant on all arcs, required to calculate the travel time from distance

995 / 5,000
实例的格式如下：

###对于每个位置，实例提供：
-StringId 作为唯一标识符
-Type 表示位置的功能，即
---d：仓库
---f：充电站
---c：客户位置
-x、y 为坐标（距离假定为欧几里得）
-demand 指定所需的货运能力数量
-ReadyTime 和 DueDate 是时间窗口的开始和结束（允许等待）
-ServiceTime 表示在客户处进行装载操作的全部时间

###对于电动汽车（所有车辆相同）：
-“Q 车辆油箱容量”：可用能量单位
-“C 车辆载重量”：可用于货物的单位
-“r 燃油消耗率”：行驶一个单位距离时电池容量的减少量
-“g 逆加油率”：充电一个单位能量所需的时间单位
-“v 平均速度”：假定在所有弧上都是常数，需要根据距离计算行驶时间