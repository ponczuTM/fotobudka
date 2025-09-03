const express = require("express");
const path = require("path");
const fs = require("fs");
const os = require("os");
const { exec } = require("child_process");

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files from the public folder
app.use(express.static(path.join(__dirname, "public")));

// Home route
app.get("/", (req, res) => {
  res.send(
    '<h1>Image Server</h1><img src="/photo/screenshot.png" alt="Example Image">'
  );
});

// Download route
app.get("/download", (req, res) => {
  const fileName = "screenshot.png";
  const filePath = path.join(__dirname, "public", "photo", fileName);

  if (!fs.existsSync(filePath)) {
    return res.status(404).send(`Plik ${fileName} nie istnieje.`);
  }

  res.download(filePath, fileName, (err) => {
    if (err) {
      console.error("Błąd przy pobieraniu:", err.message);
      res.status(500).send("Błąd podczas pobierania pliku.");
    }
  });
});

// Print route
app.get("/print", (req, res) => {
  const fileName = "screenshot.png";
  const filePath = path.join(__dirname, fileName);

  if (!fs.existsSync(filePath)) {
    return res.status(404).send(`Plik ${fileName} nie istnieje.`);
  }

  let printCommand;
  const platform = os.platform();

  if (platform === "win32") {
    printCommand = `Start-Process -FilePath "${filePath}" -Verb Print`;
    exec(`powershell -Command "${printCommand}"`, handleResult);
  } else if (platform === "darwin" || platform === "linux") {
    printCommand = `lp "${filePath}"`;
    exec(printCommand, handleResult);
  } else {
    return res.status(500).send(`co to za system ${platform}`);
  }

  function handleResult(err, stdout, stderr) {
    if (err) {
      console.error("Błąd podczas drukowania:", err.message);
      return res.status(500).send("Błąd podczas drukowania: " + err.message);
    }

    console.log("WYSŁANO");
    fs.unlink(filePath, (unlinkErr) => {
      if (unlinkErr) {
        console.error("Błąd usuwania pliku:", unlinkErr.message);
        return res
          .status(500)
          .send("Wydrukowano, ale błąd przy usuwaniu: " + unlinkErr.message);
      } else {
        console.log("screenshot.png usunięty.");
        return res.send("Wydrukowano i usunięto screenshot.png.");
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
