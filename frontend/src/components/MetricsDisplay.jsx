export default function MetricsDisplay({ mape }) {
  return <p className="mt-4">MAPE: {mape.toFixed(2)}%</p>;
}
