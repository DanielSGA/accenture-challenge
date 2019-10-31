import React from 'react';
import temp from './temp.jpg';
import './styles/bootstrap.min.css'
import { Nav } from './components/nav'
import { Card } from './components/card'
import './App.css';
import {InputGroup, FormControl, Button, Row, Col, Image, Container} from 'react-bootstrap';

function App() {
  return (
    <div className="App">
      <header className="App-header"> 

        <div className="headtitle"> Plate Search 🚘</div>

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
              <Card img={temp}></Card>
            </Col>
            <Col>
              <Card img={temp}></Card>
            </Col>
            <Col>
              <Card img={temp}></Card>
            </Col>
            <Col className="right">
              <Card img={temp}></Card>
            </Col>
          </Row>
          
          <Row>
            <Col>
              <Card img={temp}></Card>
            </Col>
            <Col>
              <Card img={temp}></Card>
            </Col>
            <Col>
              <Card img={temp}></Card>
            </Col>
            <Col className="right">
              <Card img={temp}></Card>
            </Col>
          </Row>
        </Container>

      </header>
    </div>
    
  );
}

export default App;
