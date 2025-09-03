const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

const fileName = 'screenshot.png';
const filePath = path.join(__dirname, 'public', 'photo', fileName);

if (!fs.existsSync(filePath)) {
  console.error(`Plik ${fileName} nie istnieje w /public/photo.`);
  process.exit(1);
}

const platform = os.platform();

function findDefaultPrinter(callback) {
  if (platform === 'win32') {
    exec('powershell -Command "Get-Printer | Where-Object { $_.Default -eq $true } | Select-Object -First 1 -ExpandProperty Name"', (err, stdout) => {
      if (err || !stdout.trim()) return callback(null);
      callback(stdout.trim());
    });
  } else if (platform === 'darwin' || platform === 'linux') {
    exec('lpstat -d', (err, stdout) => {
      if (err) return callback(null);
      const defaultPrinter = stdout.split(': ')[1]?.trim();
      callback(defaultPrinter || null);
    });
  } else {
    console.error(` ${platform}`);
    process.exit(1);
  }
}

function printToPrinter(printerName) {
  let printCommand;
  if (platform === 'win32') {
    printCommand = `Start-Process -FilePath "${filePath}" -Verb Print -ArgumentList '/d:"${printerName}"'`;
    exec(`powershell -Command "${printCommand}"`, handleResult);
  } else {
    printCommand = `lp -d "${printerName}" "${filePath}"`;
    exec(printCommand, handleResult);
  }
}

function handleResult(err, stdout, stderr) {
  if (err) {
    console.error('Błąd podczas drukowania:', err.message);
    return;
  }

  console.log('WYSŁANO');
  fs.unlink(filePath, (unlinkErr) => {
    if (unlinkErr) {
      console.error('Błąd usuwania pliku:', unlinkErr.message);
    } else {
      console.log('screenshot.png usunięty.');
    }
  });
}

findDefaultPrinter((printer) => {
  if (!printer) {
    console.error('Nie znaleziono domyślnej drukarki.');
    process.exit(1);
  } else {
    console.log(`Znaleziono domyślną drukarkę: ${printer}`);
    printToPrinter(printer);
  }
});
