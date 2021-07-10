use std::cmp::{Eq, Ord, Ordering, PartialOrd};
use std::collections::{BinaryHeap, HashMap};

/// 起点，v, 加入最小堆。
///
/// 在最小堆中找到下一个总开销最小的顶点弹出。
/// 保存下一个开销最小的顶点以及路径(根据存储的当前顶点)以及总开销。保存到结果集hashmap中。
/// 查找新加的开销最小的顶点的相邻点（没有在结果集中(在结果集中的必然已经是最小的了，不需要比较大小)）加入到最小堆中。
///
/// 结果为： 所有顶点V，以及V对应的路径，以及V对应的开销。hashmap
///

pub struct Graph {
    pub vertices: Vec<i32>,
    /// index: start vertex and many end vertices, every vertex: value.0: end vertex, value.1: cost value
    pub edge: Vec<Vec<(i32, i32)>>,
}

impl Graph {
    fn new() -> Graph {
        Graph {
            vertices: Vec::new(),
            edge: Vec::new(),
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct DijkstraRes {
    pub paths: HashMap<i32, Vec<i32>>, // end vertex and path
    pub costs: Vec<i32>,               // end vertex and cost
}

impl DijkstraRes {
    fn new() -> DijkstraRes {
        DijkstraRes {
            paths: HashMap::new(),
            costs: Vec::new(),
        }
    }
}

#[derive(Debug, Default)]
struct NextVertex {
    before_cost: i32,
    cost: i32,
    before: i32,
    next: i32,
}

impl NextVertex {
    fn new() -> NextVertex {
        NextVertex {
            before_cost: 0,
            cost: 0,
            before: 0,
            next: 0,
        }
    }
}

impl PartialEq for NextVertex {
    fn eq(&self, other: &Self) -> bool {
        (self.cost + self.before_cost).eq(&(other.cost + other.before_cost))
    }
}

impl PartialOrd for NextVertex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (other.cost + other.before_cost).partial_cmp(&(self.cost + self.before_cost))
    }
}

impl Ord for NextVertex {
    fn cmp(&self, other: &Self) -> Ordering {
        (other.cost + other.before_cost).cmp(&(self.cost + self.before_cost))
    }
}

impl Eq for NextVertex {
    // fn eq(&self, other: &Self) -> bool {
    //     self.cost.eq(&other.cost)
    // }
}

/// return start vertex to every vertex cost.
pub fn dijkstra(graph: Graph, start_vertex: i32) -> Option<DijkstraRes> {
    if start_vertex as usize >= graph.vertices.len() {
        return None;
    }

    // init dijkstra result
    let mut dijkstra_res = DijkstraRes::new();
    for i in 0..graph.vertices.len() as i32 {
        if i == start_vertex {
            dijkstra_res.paths.insert(i, vec![start_vertex]);
        } else {
            dijkstra_res.paths.insert(i, vec![0]);
        }
    }
    dijkstra_res.costs.resize(graph.vertices.len(), i32::MAX);
    dijkstra_res.costs[start_vertex as usize] = 0;

    let mut next_vertex = NextVertex::new();
    next_vertex.before = start_vertex;

    // construct min heap
    let mut next_min_heap = BinaryHeap::new();
    graph.edge.get(next_vertex.before as usize).map(|x| {
        // 下一个邻接点
        for next_cost in x {
            if let Some(path) = dijkstra_res.paths.get(&next_cost.0) {
                // 没有经过next_cost.0这个顶点
                if path.len() <= 1 {
                    next_min_heap.push(NextVertex {
                        before_cost: dijkstra_res.costs[start_vertex as usize],
                        before: start_vertex,
                        next: next_cost.0,
                        cost: next_cost.1,
                    });
                }
            }
        }
    });

    while !next_min_heap.is_empty() {
        if let Some(next_vertex) = next_min_heap.pop() {
            let new_cost = dijkstra_res.costs[next_vertex.before as usize] + next_vertex.cost;
            if new_cost < dijkstra_res.costs[next_vertex.next as usize] {
                dijkstra_res.costs[next_vertex.next as usize] = new_cost;

                let mut before_path = dijkstra_res
                    .paths
                    .get_mut(&next_vertex.before)
                    .unwrap()
                    .clone();
                dijkstra_res.paths.get_mut(&next_vertex.next).map(|x| {
                    x.clear();
                    x.append(&mut before_path);
                    x.push(next_vertex.next);
                });

                graph.edge.get(next_vertex.next as usize).map(|x| {
                    // 下一个邻接点
                    for next_cost in x {
                        if let Some(path) = dijkstra_res.paths.get(&next_cost.0) {
                            // 没有经过next_cost.0这个顶点
                            if path.len() <= 1 {
                                next_min_heap.push(NextVertex {
                                    before_cost: dijkstra_res.costs[next_vertex.next as usize],
                                    before: next_vertex.next,
                                    next: next_cost.0,
                                    cost: next_cost.1,
                                });
                            }
                        }
                    }
                });
            }
        }
    }
    Some(dijkstra_res)
}

#[cfg(test)]
mod test_dijkstra {
    use super::{dijkstra, Graph};
    use crate::DijkstraRes;
    use std::collections::HashMap;

    #[test]
    fn test_dijkstra_none() {
        let graph = Graph::new();
        assert_eq!(dijkstra(graph, 0), None);
    }

    #[test]
    fn test_dijkstra_one_vertex() {
        let mut graph = Graph::new();
        graph.vertices.push(0);

        let mut res = DijkstraRes {
            costs: vec![0],
            paths: HashMap::new(),
        };
        for i in 0..1 {
            if i == 0 {
                res.paths.insert(i, vec![0]);
            } else {
                res.paths.insert(i, vec![]);
            }
        }

        assert_eq!(dijkstra(graph, 0), Some(res));
    }

    #[test]
    fn test_dijkstra_one_edge() {
        let mut graph = Graph::new();
        graph.vertices.push(0);
        graph.vertices.push(1);
        graph.edge = vec![vec![(1, 10)]];

        let mut res = DijkstraRes {
            costs: vec![0, 10],
            paths: HashMap::new(),
        };
        res.paths.insert(0, vec![0]);
        res.paths.insert(1, vec![0, 1]);

        assert_eq!(dijkstra(graph, 0), Some(res));
    }

    #[test]
    fn test_dijkstra_lonely_vertex() {
        let mut graph = Graph::new();
        graph.vertices.push(0);
        graph.vertices.push(1);

        let mut res = DijkstraRes {
            costs: vec![0, i32::MAX],
            paths: HashMap::new(),
        };
        res.paths.insert(0, vec![0]);
        res.paths.insert(1, vec![0]);

        assert_eq!(dijkstra(graph, 0), Some(res));
    }

    #[test]
    fn test_dijkstra_edges() {
        let mut graph = Graph::new();
        graph.vertices.push(0);
        graph.vertices.push(1);
        graph.vertices.push(2);
        graph.vertices.push(3);

        graph.edge = vec![vec![]; 10];

        // 距离不同走开销少的。
        // 0-1: 10
        graph.edge[0].push((1, 10));
        // 1-3: 1
        graph.edge[1].push((3, 1));
        // 0-3: 30
        graph.edge[0].push((3, 30));

        // 距离相同走点数少的。
        // 0-2: 12
        graph.edge[0].push((2, 12));
        // 1-2: 2
        graph.edge[1].push((2, 2));

        let mut res = DijkstraRes {
            costs: vec![0, 10, 12, 11],
            paths: HashMap::new(),
        };
        res.paths.insert(0, vec![0]);
        res.paths.insert(1, vec![0, 1]);
        res.paths.insert(2, vec![0, 2]);
        res.paths.insert(3, vec![0, 1, 3]);

        assert_eq!(dijkstra(graph, 0), Some(res));
    }
}