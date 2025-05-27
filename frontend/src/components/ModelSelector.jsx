export default function ModelSelector({ setModel }) {
  return (
    <div>
      <label>Choose Model: </label>
      <select onChange={(e) => setModel(e.target.value)}>
        <option value="TFT">TFT</option>
        <option value="TFT+SA+MSD+AW+CSA">TFT+SA+MSD+AW+CSA</option>
      </select>
    </div>
  );
}
