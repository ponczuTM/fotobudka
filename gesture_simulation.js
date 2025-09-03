const http = require('http');
const readline = require('readline');

const gestures = {
  '1': 'NOGESTURE',
  '2': 'NOHAND',
  '3': 'V:UP',
  '4': 'FIST:UP',
  '5': 'THUMB:UP',
  '6': 'THUMB:DOWN',
  '7': 'OK:UP',
  '8': 'POINT:UP',
  '9': 'POINT:LEFT',
  '10': 'POINT:RIGHT',
  '11': 'OPENPALM:UP'
};

let currentGesture = 'NOGESTURE';

const server = http.createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/gesture') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      gesture: `X001B[GEST=${currentGesture}]`
    }));
  } else {
    res.writeHead(404);
    res.end();
  }
});

server.listen(3000, () => {
  console.log('Serwer nasłuchuje na http://localhost:3000/gesture');
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

console.log('Wybierz numer gestu (1-11):');
console.table(gestures)

rl.on('line', (input) => {
  if (gestures[input]) {
    currentGesture = gestures[input];
    console.log(`Ustawiono gest: ${currentGesture}`);
  } else {
    console.log('Nieprawidłowy numer gestu. Wprowadź 1-11.');
  }
});
