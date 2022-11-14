// Minimal webpack config to package a minified JS bundle (including
// dependencies) for execution in a <script> tag in the browser.
module.exports = {
  entry: './index.js',
  output: {
    path: __dirname,  // Directory containing this webpack.config.js file.
    filename: 'd3.js',
    // Arbitrary; many module formats could be used, just keeping Universal
    // Module Definition as it's the same as what we used in a previous
    // version.
    libraryTarget: 'umd',
  },
};
