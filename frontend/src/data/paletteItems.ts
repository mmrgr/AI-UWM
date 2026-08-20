import type { PaletteItem } from '../types/nodes';

export const paletteItems: PaletteItem[] = [
  { nodeType: 'water_resource', label: '水资源', category: '供水系统', description: '水库、河流、地下水或海水' },
  { nodeType: 'supply_conduit', label: '原水输水', category: '供水系统', description: '原水管渠与泵站' },
  { nodeType: 'wtw', label: '给水处理厂', category: '供水系统', description: '净水与药剂处理' },
  { nodeType: 'trunk_main', label: '清水干管', category: '供水系统', description: '清水输送主干线' },
  { nodeType: 'service_reservoir', label: '服务水池', category: '供水系统', description: '调蓄与消毒' },
  { nodeType: 'distribution_main', label: '配水管网', category: '供水系统', description: '用户供水与漏损' },
  { nodeType: 'local_area', label: '城市区域', category: '城市区域', description: '人口、需求与地表' },
  { nodeType: 'data_center', label: 'AI算力中心', category: 'AI基础设施', description: 'IT负荷、PUE、冷却与水源组合' },
  { nodeType: 'reuse', label: '回用设施', category: '回用系统', description: '雨水、灰水或中心回用' },
  { nodeType: 'sewer', label: '排水管网', category: '排水与污水', description: '生活、雨水或合流排水' },
  { nodeType: 'wwtw', label: '污水处理厂', category: '排水与污水', description: '污染物去除与资源回收' },
  { nodeType: 'receiving_water', label: '受纳水体', category: '排水与污水', description: '河流、湖泊或海域' },
];
