import {Network} from '../network';
import {Sample} from '../types/Sample';

function arraysEqual(a: number[], b: number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export function accuracy(network: Network, data: Sample[]): number {
  const correct = data.reduce((acc, curr) => {
    const pred = network.predict(curr.data);
    const max = Math.max(...pred);
    const result = pred.map(v => (v === max ? 1 : 0));
    if (arraysEqual(result, curr.target)) {
      acc += 1;
    }
    return acc;
  }, 0)

  return correct / data.length
}
