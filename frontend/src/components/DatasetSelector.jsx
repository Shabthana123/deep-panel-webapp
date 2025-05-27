export default function DatasetSelector({ setFile }) {
  return (
    <div>
      <label>Select CSV Dataset: </label>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
    </div>
  );
}
