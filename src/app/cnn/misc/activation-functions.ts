import {map} from 'mathjs';

export function sigmoid(z: number[]): number[] {
  return map(z, (x) => 1/(1 + Math.exp(-x)))
}

export function softmax(z: number[]): number[] {
  const max = Math.max(...z)

  // subtracting for stability
  const expZ = z.map(x => Math.exp(x - max))
  const sum = expZ.reduce((acc, curr) => acc + curr, 0)
  return expZ.map(x => x / sum)
}

export function relu(z: number[]): number[] {
  return z.map((x) => Math.max(0, x))
}
