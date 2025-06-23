function trocaTema() {
  const meuIcon = document.querySelector('.meuIcon');
  const meuEstilo = document.getElementById('meuEstilo1');
  let estiloAtual = 'dayTheme.css';

  function handler() {
    meuIcon.removeEventListener('click', handler);

    setTimeout(function() {
      requestAnimationFrame(function() {
        if (estiloAtual === 'dayTheme.css') {
          meuEstilo.href = 'assets/css/nightTheme.css';
          estiloAtual = 'nightTheme.css';
        }else{
          meuEstilo.href = 'assets/css/dayTheme.css';
          estiloAtual = 'dayTheme.css';
        }
        meuIcon.addEventListener('click', handler);
      });
    }, null);
  }
  meuIcon.addEventListener('click', handler);
}
trocaTema();
