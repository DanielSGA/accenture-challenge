import React from 'react';
import { Image } from 'react-bootstrap';

export const Card = ({img, name, color})=>{
    return(
        <div className="card">
        <img src={img} alt="temp" className="imag"/>
          <div className="container">
            <h4><b>{name}</b></h4>
            <span className={"dot " + color}></span>
          </div>
        </div>
    )
}

export default Card;
