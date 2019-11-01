import React from 'react';
import temp from './img/temp.jpg';
import './styles/bootstrap.min.css'
import { Card } from './components/card'
import './App.css';
import {InputGroup, FormControl, Button, Row, Col, Image, Container} from 'react-bootstrap';
// import response from './server.js'
import platesArr from './components/images'

function App() {
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
              />
              <InputGroup.Append>
                <Button variant="outline-secondary">Search</Button>
              </InputGroup.Append>
            </InputGroup>
          </Col>
        </Row>

        <Container>
          <Row>
            <Col>
              <Card img={platesArr[0].imag} name={platesArr[0].name} color={platesArr[0].color}></Card>
            </Col>
            <Col>
              <Card img={platesArr[1].imag} name={platesArr[1].name} color={platesArr[1].color}></Card>
            </Col>
            <Col>
              <Card img={platesArr[2].imag} name={platesArr[2].name} color={platesArr[2].color}></Card>
            </Col>
            <Col className="right">
              <Card img={platesArr[4].imag} name={platesArr[4].name} color={platesArr[4].color}></Card>
            </Col>
          </Row>
          <Row>
            <Col>
              <Card img={platesArr[3].imag} name={platesArr[3].name} color={platesArr[3].color}></Card>
            </Col>
            <Col>
            </Col>
            <Col>
            </Col>
            <Col>
            </Col>
          </Row>
        </Container>
      </header>
    </div>
  );
}

export default App;
