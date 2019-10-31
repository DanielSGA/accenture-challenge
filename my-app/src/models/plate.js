const mongoose = require('mongoose')

const plateSchema = new mongoose.Schema({
    image: {type: Image},
    name: {type: String},
    color: {type: String},
  })

const Plate = mongoose.model('Plate',plateSchema,'plate')
module.exports = Plate