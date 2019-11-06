import React from 'react';
import temp from './img/temp.jpg';
import './styles/bootstrap.min.css'
import { Card } from './components/card'
import './App.css';
import {InputGroup, FormControl, Button, Row, Col, Image, Container} from 'react-bootstrap';
// import response from './server.js'
import platesArr from './components/images'



class App extends React.Component {

  constructor(props){
    super(props)
    this.state = {inputValue: "",temparr : platesArr}
  }

  updateValue=(event)=>{
    this.setState({
      inputValue: event.target.value
    })
  }

   filterPlates=()=> {
    console.log(this.state.inputValue)
    console.log(platesArr[0].name)
    let item = []
    for(var i = 0; i < 4; i++){
      if(this.state.inputValue === platesArr[i].name)
      {

         item = platesArr[i]
         this.render()
      }
    }
    this.setState({temparr:[item]})
    }

  render(){
    const temparr = this.state.temparr
    return (
      <div className="App">
        <header className="App-header"> 
          <div className="headtitle">Plate Search 🚘</div>
          <Row>
            <Col md={{ span: 6, offset: 3 }}>
              <InputGroup className="mb-3 searchbar justify-content-md-center">
                <FormControl
                  placeholder="Write your car's plate number"
                  aria-label="Write your car's plate number"
                  aria-describedby="basic-addon2"
                  inputRef={ref => { this.myInput = ref; }}
                  onChange={this.updateValue}
                />
                <InputGroup.Append>
                  <Button variant="outline-secondary" onClick={this.filterPlates}>Search</Button>
                </InputGroup.Append>
              </InputGroup>
            </Col>
          </Row>
  
          <Container>
            <Row>

              {
                temparr.map(item=>{
                return (<Col md={{ span: 2, offset: 1 }}>
                  <Card img={item.imag} name={item.name} color={item.color}></Card>
                </Col>)
              })
              }
            </Row>
            
          </Container>
        </header>
      </div>
    );
  }
}

export default App;
